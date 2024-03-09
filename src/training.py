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

from dataloader import DrugTargetData, ModuleDataProcessor
from model import BaseTargetIdentifier, BaseLightningTargetIdentifier
from puupl import training as puupl_training
from plot import umap, plot_embedding_distribution


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


def normalise_data(train_raw, val_raw, train_genes, val_genes, test_genes, acmg_raw, pfam_raw, config, norm,
                   model_type="mlp"):
    hparams = config['hyperparameters']

    if norm:
        val_norm = val_raw.iloc[:, :-1].values

        gene_names_train = train_genes
        train_norm = train_raw.iloc[:, :-1].values

        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(train_norm)

        _train = DataLoader(
            DrugTargetData(data=train_norm, labels=train_raw.iloc[:, -1].values, gene_names=gene_names_train),
            batch_size=int(hparams[model_type]['batch_size']),
            shuffle=True,
            num_workers=int(hparams[model_type]['num_workers'])
        )

        gene_names_val = val_genes
        val_norm = scaler.transform(val_norm)
        val = DataLoader(
            DrugTargetData(data=val_norm, labels=val_raw.iloc[:, -1].values, gene_names=gene_names_val),
            batch_size=int(hparams[model_type]['batch_size']),
            shuffle=False
        )

        acmg_gene_names = test_genes['acmg']
        acmg_norm = acmg_raw.iloc[:, :-1].values
        acmg_norm = scaler.transform(acmg_norm)
        acmg_test = DataLoader(
            DrugTargetData(data=acmg_norm, labels=acmg_raw.iloc[:, -1].values, gene_names=acmg_gene_names,
                           test_source='acmg'),
            batch_size=len(acmg_raw),
            shuffle=False
        )

        pfam_gene_names = test_genes['pfam']
        pfam_norm = pfam_raw.iloc[:, :-1].values
        pfam_norm = scaler.transform(pfam_norm)
        pfam_test = DataLoader(
            DrugTargetData(data=pfam_norm, labels=pfam_raw.iloc[:, -1].values, gene_names=pfam_gene_names,
                           test_source='pfam'),
            batch_size=len(pfam_raw),
            shuffle=False
        )
    else:
        val_data = val_raw.iloc[:, :-1].values
        gene_names_train = train_genes
        train_data = train_raw.iloc[:, :-1].values

        _train = DataLoader(
            DrugTargetData(data=train_data, labels=train_raw.iloc[:, -1].values, gene_names=gene_names_train),
            batch_size=int(hparams[model_type]['batch_size']),
            shuffle=True,
            num_workers=int(hparams[model_type]['num_workers'])
        )

        gene_names_val = val_genes
        val = DataLoader(
            DrugTargetData(data=val_data, labels=val_raw.iloc[:, -1].values, gene_names=gene_names_val),
            batch_size=int(hparams[model_type]['batch_size']),
            shuffle=False
        )

        acmg_gene_names = test_genes['acmg']
        acmg_data = acmg_raw.iloc[:, :-1].values
        acmg_test = DataLoader(
            DrugTargetData(data=acmg_data, labels=acmg_raw.iloc[:, -1].values, gene_names=acmg_gene_names,
                           test_source='acmg'),
            batch_size=len(acmg_data),
            shuffle=False
        )

        pfam_gene_names = test_genes['pfam']
        pfam_data = pfam_raw.iloc[:, :-1].values
        pfam_test = DataLoader(
            DrugTargetData(data=pfam_data, labels=pfam_raw.iloc[:, -1].values, gene_names=pfam_gene_names,
                           test_source='pfam'),
            batch_size=len(pfam_data),
            shuffle=False
        )


    return _train, val, acmg_test, pfam_test


def initialise_model(train_raw, val_raw, train_genes, val_genes, test_genes, acmg_data, pfam_data, num_features,
                     config, norm):
    hyperparams = config['hyperparameters']
    _train, val, acmg_test, pfam_test = normalise_data(train_raw, val_raw, train_genes, val_genes, test_genes,
                                                       acmg_data, pfam_data, config, norm)

    train_imbalance = 1 / float(_train.dataset.label_imbalance().item())  # calculate inverse class frequency

    mlp_pytorch = BaseTargetIdentifier(config=config, num_features=num_features)
    mlp_lightning = BaseLightningTargetIdentifier(model=mlp_pytorch, config=config, imbalance=train_imbalance)

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

    return mlp_lightning, _train, val, acmg_test, pfam_test, hyperparameters, accelerator


def train(tag="Training"):
    gcp = ModuleDataProcessor.open_gc_data
    data = gcp.data
    num_features = gcp.num_features
    config = gcp.config

    acmg_data = gcp.acmg_data
    pfam_data = gcp.pfam_data

    # Note: Validation samples are clinically actionable genes as defined by the ACMG (see:
    # https://www.coriell.org/1/NIGMS/Collections/ACMG-73-Genes). We also have a broader set of clinically actionable
    # genes - with scores - from ClinGen. These are currently loaded in `get_actionable_genes` in `preprocessing.py`.
    # These need NOT yet have FDA-approved drugs.

    train_raw, val_raw = train_test_split(data, test_size=0.2, random_state=42)

    mlp_lightning, _train, val, acmg_test, pfam_test, hyperparameters, accelerator = (
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
        trainer.test(ckpt_path="best", dataloaders=acmg_test)
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


def kfold_train(data: pd.DataFrame, genes: pd.DataFrame, test_genes: Dict[str, pd.DataFrame], acmg_data: pd.DataFrame,
                pfam_data: pd.DataFrame, num_features: int, config: dict, model_type: str,
                modules: Union[str, Dict[str, bool]], norm: bool = True):
    num_splits = 5
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    if isinstance(modules, str):
        used_modules = [modules]
    else:
        used_modules = [k for k, v in modules.items() if v]

    module_str = f"{'-'.join(used_modules)}"

    group = f"distillation-1-{model_type}-{module_str}"
    if not os.path.isdir(f'checkpoints/{group}'):
        os.mkdir(f'checkpoints/{group}')
    else:
        i = 1
        while os.path.isdir(f'checkpoints/{group}'):
            i += 1
            group = f"distillation-{i}-{model_type}-{module_str}"
        os.mkdir(f'checkpoints/{group}')

    # use the new train object to train a new model
    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{num_splits}")

        # Split the data
        train_raw = data.iloc[train_indices, :]
        val_raw = data.iloc[val_indices, :]

        train_genes = genes.iloc[train_indices]
        val_genes = genes.iloc[val_indices]

        mlp_lightning, _train, val, acmg_test, pfam_test, hyperparameters, accelerator = (
            initialise_model(train_raw, val_raw, train_genes, val_genes, test_genes, acmg_data, pfam_data, num_features,
                             config, norm)
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
        trainer.test(ckpt_path="best", dataloaders=acmg_test)
        trainer.test(ckpt_path="best", dataloaders=pfam_test)

        run.finish()


def kfold_teacher(ensemble=False, **modules):
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
                umap(train_df)
                genes = preprocessor.ensg_ids
                num_features = preprocessor.num_features
                norm = preprocessor.norm
                config = preprocessor.config
                acmg_data = preprocessor.acmg_data
                pfam_data = preprocessor.pfam_data
                acmg_genes = preprocessor.acmg_ids
                pfam_genes = preprocessor.pfam_ids
                test_genes = {
                    "acmg": acmg_genes,
                    "pfam": pfam_genes
                }
                kfold_train(train_df, genes, test_genes, acmg_data, pfam_data, num_features, config,
                            model_type="teacher", modules=module, norm=norm)
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
