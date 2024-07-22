import gc

import yaml
import os
import torch
import optuna
import wandb

import utils

import lightning as pl
import pandas as pd
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union
from tqdm import tqdm

from dataloader import DrugTargetData, ModuleDataProcessor, DrugTargetVAEData
from model import BaseTargetIdentifier, BaseLightningTargetIdentifier, VariantRepresentationTargetIdentifier
from utils import df_col_to_dense
from autoencoders.vae import VAE
from puupl import training as puupl_training
# from plot import umap, plot_embedding_distribution


def tune():
    study = optuna.create_study(
        study_name="gdtp_mlp",
        direction="maximize"
    )
    n_trials = 100

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial, optimized for auROC:")
    best_trial = study.best_trial

    print("  Metric: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    if not os.path.isfile("best_hyperparameters.txt"):
        with open("best_hyperparameters.txt", "w") as f:
            f.write("Run ID\tauROC\tHyperparameters\tWidth/Depth\t#Params\n")
        run_id = 0
    else:
        with open("best_hyperparameters.txt", "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            run_id = int(last_line.split("\t")[0]) + 1

    with open("best_hyperparameters.txt", "a") as f:
        f.write(f"{run_id}\t{best_trial.value}\t{best_trial.params}\t{best_trial.params}\t"
                f"{best_trial.user_attrs['width_depth_ratio']}\t{best_trial.user_attrs['total_params']}\n")


def objective(trial: optuna.trial.Trial) -> float:
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    config = config['hyperparameters']

    config['mlp']['depth'] = trial.suggest_int('depth', 2, 54, step=4)
    config['mlp']['width'] = trial.suggest_int('width', 4, 260, step=16)
    config['mlp']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    config['mlp']['threshold'] = trial.suggest_float('threshold', 0.2, 0.8, step=0.05)
    config['mlp']['weight_decay'] = trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.01, 0.1])

    trainer = train(tag="Tuning")

    width_depth_ratio = config['mlp']['width'] / config['mlp']['depth']
    total_params = sum(p.numel() for p in trainer.model.parameters())

    trial.set_user_attr('width_depth_ratio', width_depth_ratio)
    trial.set_user_attr('total_params', total_params)

    return trainer.callback_metrics["val_auroc"].item()


def normalise_data(train_raw, val_raw, train_genes, val_genes, test_genes, test_raw, config, module_str):
    hparams = config['hyperparameters']

    train_raw['pathogenicity'] = train_raw['pathogenicity'].apply(df_col_to_dense)
    print("\n\n\n")
    print("Val raw shape: ", val_raw['pathogenicity'].shape)
    print("\n\n\n")
    print("Val raw: ", val_raw['pathogenicity'])
    val_raw['pathogenicity'] = val_raw['pathogenicity'].apply(df_col_to_dense)

    # train_raw['pathogenicity'] = train_raw['pathogenicity'].transform(df_col_to_dense)
    # val_raw['pathogenicity'] = val_raw['pathogenicity'].transform(df_col_to_dense)

    # chunk_size = 1000  # Adjust based on your available memory
    # for start in tqdm(range(0, len(train_raw), chunk_size)):
    #     end = start + chunk_size
    #     train_raw.iloc[start:end, train_raw.columns.get_loc('pathogenicity')] = train_raw.iloc[start:end,
    #                                                                             train_raw.columns.get_loc(
    #                                                                                 'pathogenicity')].apply(
    #         df_col_to_dense)

    # Normalize the training data
    train_norm_raw = train_raw.iloc[:, :-1].values
    train_norm = np.vstack(train_norm_raw[:, 0])
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_norm)

    val_norm_raw = val_raw.iloc[:, :-1].values
    val_norm = np.vstack(val_norm_raw[:, 0])
    val_norm = scaler.transform(val_norm)

    # for start in tqdm(range(0, len(val_raw), chunk_size)):
    #     end = start + chunk_size
    #     val_raw.loc[start:end, 'pathogenicity'] = val_raw.loc[start:end, 'pathogenicity'].apply(df_col_to_dense)
    #     val_raw.iloc[start:end, val_raw.columns.get_loc('pathogenicity')] = val_raw.iloc[start:end,
    #                                                                             val_raw.columns.get_loc(
    #                                                                                 'pathogenicity')].apply(
    #         df_col_to_dense)

    # train_tst = val_raw.iloc[-100:, :]
    #
    # for row in train_tst.iterrows():
    #     # find which vectors in the pathogenicity column contain non-zero values
    #     vec = row[1]['pathogenicity']
    #     print('joe')

    drug_target_train_data = {
        'data': train_norm,
        'labels': train_raw.iloc[:, -1].values,
        'gene_names': train_genes
    }

    drug_target_val_data = {
        'data': val_norm,
        'labels': val_raw.iloc[:, -1].values,
        'gene_names': val_genes
    }

    drug_target_test_data = {}
    for key, raw in test_raw.items():
        test_norm_raw = raw.iloc[:, :-1].values
        test_norm = np.vstack(test_norm_raw[:, 0])
        test_norm = scaler.transform(test_norm)
        drug_target_test_data[key] = {
            'data': test_norm,
            'labels': raw.iloc[:, -1].values,
            'gene_names': test_genes[key],
            'test_source': key
        }

    if module_str == "pvc":
        _train = DataLoader(
            DrugTargetVAEData(
                drug_target_train_data,
                reduct_dim=hparams['pathogenicity_embedding']['io_dim']
            ),
            batch_size=hparams['pathogenicity_embedding']['batch_size'],
            shuffle=True,
            num_workers=hparams['pathogenicity_embedding']['num_workers']
        )

        val = DataLoader(
            DrugTargetVAEData(
                drug_target_val_data,
                reduct_dim=hparams['pathogenicity_embedding']['io_dim']
            ),
            batch_size=hparams['pathogenicity_embedding']['batch_size'],
            shuffle=False,
            num_workers=hparams['pathogenicity_embedding']['num_workers']
        )

        test = {}
        for key, data in drug_target_test_data.items():
            test[key] = DataLoader(
                DrugTargetVAEData(
                    data,
                    reduct_dim=hparams['pathogenicity_embedding']['io_dim']
                ),
                batch_size=hparams['pathogenicity_embedding']['batch_size'],
                shuffle=False,
                num_workers=hparams['pathogenicity_embedding']['num_workers']
            )
    else:
        _train = DataLoader(
            DrugTargetData(
                data=train_norm,
                labels=train_raw.iloc[:, -1].values,
                gene_names=train_genes
            ),
            batch_size=int(hparams['mlp']['batch_size']),
            shuffle=True,
            num_workers=int(hparams['mlp']['num_workers'])
        )

        val = DataLoader(
            DrugTargetData(
                data=scaler.transform(val_raw.iloc[:, :-1].values),
                labels=val_raw.iloc[:, -1].values,
                gene_names=val_genes
            ),
            batch_size=int(hparams['mlp']['batch_size']),
            shuffle=False,
            num_workers=int(hparams['mlp']['num_workers'])
        )

        test = {}
        for key, raw in test_raw.items():
            normed = scaler.transform(raw.iloc[:, :-1].values)
            test[key] = DataLoader(
                DrugTargetData(
                    data=normed,
                    labels=raw.iloc[:, -1].values,
                    gene_names=test_genes[key],
                    test_source=key
                ),
                batch_size=len(raw),
                shuffle=False,
                num_workers=int(hparams['mlp']['num_workers'])
            )

    # if isinstance(_train.dataset, DrugTargetData):
    label_imbalance = _train.dataset.label_imbalance().item()

    # else:
    #     train_dtd = DataLoader(
    #         DrugTargetData(
    #             data=train_norm,
    #             labels=train_raw.iloc[:, -1].values,
    #             gene_names=gene_names_train
    #         ),
    #         batch_size=int(hparams['mlp']['batch_size']),
    #         shuffle=True,
    #         num_workers=int(hparams['mlp']['num_workers'])
    #     )
    #     label_imbalance = train_dtd.dataset.label_imbalance().item()

    return _train, val, test, label_imbalance


def initialise_model(train_raw, val_raw, train_genes, val_genes, test_genes, test, num_features, config, module_str):
    hyperparams = config['hyperparameters']
    _train, val, test, train_imbalance = normalise_data(train_raw, val_raw, train_genes, val_genes, test_genes, test,
                                                        config, module_str)

    mlp_pytorch = BaseTargetIdentifier(config=config, num_features=num_features)

    if module_str == "pvc":
        vae = VAE(input_dim=hyperparams['pathogenicity_embedding']['io_dim'],
                  latent_dim=hyperparams['pathogenicity_embedding']['latent_dim'])
        model = VariantRepresentationTargetIdentifier(
            vae, mlp_pytorch, config, num_features,
            latent_dim=hyperparams['pathogenicity_embedding']['latent_dim'],
            imbalance=train_imbalance
        )
    else:
        model = BaseLightningTargetIdentifier(model=mlp_pytorch, config=config, imbalance=train_imbalance)

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    hyperparameters = dict(
        depth=hyperparams['mlp']['depth'],
        lr=hyperparams['mlp']['lr_start'],
        batch_size=hyperparams['mlp']['batch_size'],
        optimizer=hyperparams['mlp']['optimizer'],
        epochs=hyperparams['mlp']['epochs'],
        dropout=hyperparams['mlp']['dropout'],
        width=hyperparams['mlp']['width'],
        weight_decay=hyperparams['mlp']['weight_decay'],
    )

    return model, _train, val, test, hyperparameters, accelerator


def train(tag="Training"):
    gcp = ModuleDataProcessor.open_gc_data
    data = gcp.data
    num_features = gcp.num_features
    config = gcp.config

    acmg_data = gcp.acmg_data
    pfam_data = gcp.pfam_data

    train_raw, val_raw = train_test_split(data, test_size=0.2, random_state=42)

    mlp_lightning, _train, val, pfam_test, hyperparameters, accelerator = (
        initialise_model(train_raw, val_raw, acmg_data, pfam_data, num_features, config)
    )

    if tag == "Standard Training":
        wandb_logger = WandbLogger(
            project="drug-target-prediction",
            tags=[f"depth{config['mlp']['depth']}-nn"],
            log_model="all"
        )
        wandb_logger.log_hyperparams(hyperparameters)
        run_name = wandb_logger.experiment.name
        checkpoint_callback = ModelCheckpoint(
            monitor='epoch',
            dirpath='checkpoints',
            filename=f'{run_name}' + '-{epoch:02d}-{val_auroc:.2f}',
            save_top_k=1,
            mode='max',
        )

        utils.set_seed(42)
        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )
        trainer.fit(mlp_lightning, _train, val)
        trainer.test(ckpt_path="best", dataloaders=pfam_test)
        return trainer
    elif tag == "PUUPL Training":
        puupl_training(train=train, val=val, config=config)
    elif tag == "Tuning":
        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(mlp_lightning, train, val)

        return trainer
    else:
        raise ValueError("Invalid tag. Pick from 'Standard Training', 'PUUPL Training' or 'Tuning'")


def kfold_train(
        data: pd.DataFrame,
        genes: pd.DataFrame,
        test_genes: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        num_features: int,
        config: dict,
        model_type: str,
        modules: Union[str, Dict[str, bool]],
        norm: bool = True
):
    num_splits = 5
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    if isinstance(modules, str):
        used_modules = [modules]
    else:
        used_modules = [k for k, v in modules.items() if v]

    module_str = f"{'-'.join(used_modules)}"

    group = f"distillation-1-{model_type}-{module_str}"
    if not os.path.isdir(f'checkpoints/{group}'):
        os.mkdir('checkpoints')
        os.mkdir(f'checkpoints/{group}')
    else:
        i = 1
        while os.path.isdir(f'checkpoints/{group}'):
            i += 1
            group = f"distillation-{i}-{model_type}-{module_str}"
        os.mkdir(f'checkpoints/{group}')

    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{num_splits}")
        
        # Split the data
        train_raw = data.iloc[train_indices, :]
        val_raw = data.iloc[val_indices, :]

        train_genes = genes.iloc[train_indices]
        val_genes = genes.iloc[val_indices]

        mlp_lightning, _train, val, test, hyperparameters, accelerator = initialise_model(
            train_raw,
            val_raw,
            train_genes,
            val_genes,
            test_genes,
            test_data,
            num_features,
            config,
            module_str
        )

        run = wandb.init(
            project="drug-target-prediction",
            tags=[f"distillation-fold{fold + 1}"],
            config=hyperparameters,
            group=f"{group}"
        )

        run_name = wandb.run.name
        checkpoint_callback = ModelCheckpoint(
            monitor='epoch',
            dirpath=f'checkpoints/{group}',
            filename=f'{run_name}' + '-{epoch:02d}-{val_auroc:.2f}-' + f'fold{fold}',
            save_top_k=1,
            mode='max'
        )

        utils.set_seed(42)
        trainer = pl.Trainer(
            max_epochs=int(config['hyperparameters']['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=WandbLogger(wandb.run),
            callbacks=[checkpoint_callback]
        )

        trainer.fit(mlp_lightning, _train, val)
        trainer.test(ckpt_path="best", dataloaders=test["pfam"])
        trainer.test(ckpt_path="best", dataloaders=test["rcnt"])
        trainer.test(ckpt_path="best", dataloaders=test["pharos"])

        run.finish()


def kfold_teacher(ensemble=False, **modules):
    pl.seed_everything(42)

    print("Training teacher model...\n")

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)

    modules = {
        "gc": gc,
        "go": go,
        "pvc": pvc,
        "psc": psc
    }

    print(f"Training teacher model with {' '.join([k for k, v in modules.items() if v])} modules...\n")

    data = ModuleDataProcessor(gc, go, pvc, psc).process()

    if not ensemble:
        for module, preprocessor in data.items():
            if modules[module]:
                train_df = preprocessor.data
                # umap(train_df)
                genes = preprocessor.ensg_ids
                num_features = preprocessor.num_features
                norm = preprocessor.norm
                config = preprocessor.config
                pfam_data = preprocessor.pfam_data
                pfam_genes = preprocessor.pfam_ids
                rcnt_data = preprocessor.rcnt_data
                rcnt_genes = preprocessor.rcnt_ids
                pharos_data = preprocessor.pharos_data
                pharos_genes = preprocessor.pharos_ids
                test_data = {
                    "pfam": pfam_data,
                    "rcnt": rcnt_data,
                    "pharos": pharos_data
                }
                test_genes = {
                    "pfam": pfam_genes,
                    "rcnt": rcnt_genes,
                    "pharos": pharos_genes
                }
                kfold_train(train_df, genes, test_genes, test_data, num_features, config, model_type="teacher",
                            modules=module, norm=norm)
    else:
        # TODO: Implement ensemble training
        pass


def kfold_student(ensemble=False, **modules):
    """
    Training a student model by distilling from a teacher model where the teacher model is used to provide pseudo-labels
    for the unlabeled data used by the student model.

    Firstly, we will run inference over the unlabelled data using the teacher model to define the pseudolabels. Then,
    we add the pseudolabels to the training data and train the student model on the combined dataset.
    """
    # TODO: Connect this to the teacher to learn end-to-end
    print("Training student model by distillation from teacher model...\n")

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)

    modules = {
        "gc": gc,
        "go": go,
        "pvc": pvc,
        "psc": psc
    }

    print(f"Training teacher model with {' '.join([k for k, v in modules.items() if v])} modules...\n")

    data = ModuleDataProcessor(gc, go, pvc, psc).process()

    if not ensemble:
        for module, preprocessor in data.items():
            train_df = preprocessor.data
            num_features = preprocessor.num_features
            config = preprocessor.config

            labels = train_df.iloc[:, -1].values

            unlabelled_data = data[data.iloc[:, -1] == 0]
            U = np.where(labels == 0)[0]

            hyperparams = config['hyperparameters']

            val = None
            unlabelled_dl = normalise_data(unlabelled_data, val, hyperparams)
            unlabelled_tensor = unlabelled_dl.dataset.features

            teacher_model = BaseTargetIdentifier(config=hyperparams, num_features=num_features)
            raw_state_dict = torch.load(
                "checkpoints/distillation-3-teacher/swift-jazz-179-epoch=99-val_auroc=0.84-fold0.ckpt"
            )['state_dict']

            state_dict = {k.replace("model.", ""): v for k, v in raw_state_dict.items()}
            teacher_model.load_state_dict(state_dict)
            teacher_model.eval()

            logits, probas, bin_preds = teacher_model(unlabelled_tensor)

            delta = 0.25  # threshold for pseudo-labeling

            pseudo_labels = torch.full_like(probas, -1)

            pos_count = 0
            neg_count = 0
            for i, proba in enumerate(probas):
                if proba >= hyperparams['mlp']['threshold']:
                    pseudo_labels[i] = proba
                    pos_count += 1
                elif proba <= hyperparams['mlp']['threshold'] - delta:
                    pseudo_labels[i] = proba
                    neg_count += 1
            unlabelled_count = len(probas) - pos_count - neg_count

            # plot.plot_kde(pseudo_labels)

            data.iloc[U, -1] = pseudo_labels.detach().numpy()

            data = data[data.iloc[:, -1] != -1]

            kfold_train(data, num_features, hyperparams, model_type="student", modules=module)
    else:
        # TODO: Implement ensemble training
        pass
