import gc

import yaml
import os
import torch
import optuna
import wandb

import utils

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import pickle as pkl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, Optional
from tqdm import tqdm

from dataloader import DrugTargetData, ModuleDataProcessor, VarformerDataset
from models.target_identifier import MultiModalTargetIdentifier
from models.lightning import (MLPLightningTargetIdentifier, ShardedVarformerLightningTargetIdentifier,
                              MultiModalLightningTargetIdentifier)


def tune():
    study = optuna.create_study(
        study_name="gdtp_varformer",
        direction="maximize"
    )
    n_trials = 1000

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial, optimized for auROC:")
    best_trial = study.best_trial

    print("  Metric: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


def objective(trial: optuna.trial.Trial) -> float:
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    varformer_usage = config['hyperparameters']['varformer_usage']
    if not varformer_usage:
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
    else:
        config['hyperparameters']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        # config['hyperparameters']['threshold'] = trial.suggest_float('threshold', 0.2, 0.8, step=0.1)
        # lr_start = trial.suggest_float('lr_start', 1e-6, 1e-4, log=True)
        # config['hyperparameters']['lr_start'] = lr_start
        # config['hyperparameters']['lr_end'] = trial.suggest_float('lr_end', 1e-7, lr_start, log=True)
        # config['hyperparameters']['weight_decay'] = trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.01, 0.1])
        config['hyperparameters']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 4])
        config['hyperparameters']['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers', [1, 2, 4, 6])
        config['hyperparameters']['nhead'] = trial.suggest_categorical('nhead', [1, 2, 4, 6])
        config['hyperparameters']['width'] = trial.suggest_categorical('width', [64, 128, 256, 512])
        config['hyperparameters']['T0'] = trial.suggest_categorical('T0', [50, 100, 200, 400, 1000])

        hyperparameters = dict(
            depth=config['hyperparameters']['num_layers'],
            # lr_start=config['hyperparameters']['lr_start'],
            # lr_end=config['hyperparameters']['lr_end'],
            batch_size=config['hyperparameters']['batch_size'],
            optimizer=config['hyperparameters']['optimizer'],
            epochs=config['hyperparameters']['epochs'],
            dropout=config['hyperparameters']['dropout'],
            width=config['hyperparameters']['width'],
            # weight_decay=config['hyperparameters']['weight_decay'],
            # threshold=config['hyperparameters']['threshold'],
            num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
            nhead=config['hyperparameters']['nhead'],
            T0=config['hyperparameters']['T0']
        )

        module_str = "pvc"

        # Initialize a wandb run
        run = wandb.init(
            project="drug-target-prediction",
            tags=["hpo-varformer"],
            group="hyperparameter-tuning",
            dir="/data/scratch/bty174/genomic-drug-targeting/src/",
            config=hyperparameters
        )

        # Log the updated hyperparameters
        wandb.config.update(hyperparameters)

        if varformer_usage:
            mdp = ModuleDataProcessor(gc=False, go=False, pvc=True, psc=False)
        else:
            mdp = ModuleDataProcessor(gc=True, go=False, pvc=False, psc=False)

        gcp = mdp.open_gc_data()
        if varformer_usage:
            pvc = mdp.open_pvc_data(gc_data=gcp, tune=True)
            data = pvc.data
            num_features = gcp.num_features
            labels = data['labels']
            data.pop('labels')
        else:
            data = gcp.data
            num_features = gcp.num_features
            labels = gcp.labels

        test_data = None
        test_genes = None

        gene_ids = list(data.keys())

        # Split keys into train and test sets
        train_ids, val_ids = train_test_split(gene_ids, test_size=0.2, random_state=42)

        # Create train and test dictionaries
        train_raw = {gene_id: data[gene_id] for gene_id in train_ids}
        val_raw = {gene_id: data[gene_id] for gene_id in val_ids}

        train_genes = pd.Index(train_ids)
        val_genes = pd.Index(val_ids)

        mlp_lightning, _train, val, test, hyperparameters, accelerator = initialise_model(
            train_raw,
            val_raw,
            labels,
            train_genes,
            val_genes,
            test_genes,
            test_data,
            num_features,
            config,
            module_str
        )

        if config['hyperparameters']['wandb']:
            utils.set_seed(42)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            checkpoint_callback = ModelCheckpoint(
                monitor='val_auroc',
                dirpath="/data/scratch/bty174/genomic-drug-targeting/src/lightning_logs",
                filename='{epoch:02d}-{val_auroc:.2f}',
                save_top_k=1,
                mode='max'
            )
            if torch.cuda.device_count() > 1:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[lr_monitor],
                    precision=config['hyperparameters']['precision'],
                    strategy="ddp_find_unused_parameters_true",
                    devices=-1,
                    gradient_clip_val=1,
                    deterministic=True
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[lr_monitor, checkpoint_callback],
                    gradient_clip_val=1,
                    deterministic=True
                )
        else:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=1,
                logger=False,
                enable_checkpointing=True,
                callbacks=[checkpoint_callback]
            )

        trainer.fit(mlp_lightning, _train, val)
        run.finish()

        return trainer.callback_metrics["val_auroc"].item()


def normalise_data(train_raw, val_raw, labels, train_genes, val_genes, test_genes, test_raw, config, module_str):
    hparams = config['hyperparameters']
    train_combined = {}
    val_combined = {}
    test_combined = {}
    for module_str, train_data in train_raw.items():
        if module_str == "pvc":
            all_genes = list(labels.keys())

            drug_target_train_data = {
                'data': train_data,
                'labels': labels
            }

            drug_target_val_data = {
                'data': val_raw[module_str],
                'labels': labels
            }

            train_pvc = DataLoader(
                VarformerDataset(
                    drug_target_train_data,
                    max_variants=hparams['max_seq_len'],
                ),
                batch_size=hparams['batch_size'],
                shuffle=True,
                num_workers=hparams['num_workers']
            )

            val_pvc = DataLoader(
                VarformerDataset(
                    drug_target_val_data,
                    max_variants=hparams['max_seq_len'],
                ),
                batch_size=hparams['batch_size'],
                shuffle=False,
                num_workers=hparams['num_workers']
            )

            # if test_raw['rcnt']['pvc'] is not None:
            #     variant_counts = {}
            #     for gene in all_genes:
            #         count = 0
            #         if gene in train_data:
            #             count += train_data[gene].shape[0]
            #         if gene in val_raw:
            #             count += val_raw[gene].shape[0]
            #         if gene in test_raw['pvc']:
            #             count += test_raw['pvc'][gene].shape[0]
            #         variant_counts[gene] = count

            drug_target_test_data = {}
            for key, modalities in test_raw.items():
                drug_target_test_data[key] = {
                    'data': modalities[module_str],
                    'labels': labels,
                    'test_source': key
                }

            test_pvc = {}
            for key, test_data in drug_target_test_data.items():
                test_dataset = VarformerDataset(
                    test_data,
                    max_variants=hparams['max_seq_len'],
                    test_source=key
                )
                test_pvc[key] = DataLoader(
                    test_dataset,
                    batch_size=len(test_dataset),
                    shuffle=False,
                    num_workers=hparams['num_workers']
                )
            else:
                test = None
            train_combined[module_str] = train_pvc
            val_combined[module_str] = val_pvc
            test_combined[module_str] = test_pvc
            label_imbalance = train_pvc.dataset.label_imbalance()
        else:
            val_norm = val_raw[module_str].iloc[:, :-1].values
            train_norm = train_data.iloc[:, :-1].values

            scaler = MinMaxScaler()

            train_norm = scaler.fit_transform(train_norm)
            val_norm = scaler.transform(val_norm)

            train_loader = DataLoader(
                DrugTargetData(
                    data=train_norm,
                    labels=train_data.iloc[:, -1].values,
                    gene_names=train_genes
                ),
                batch_size=int(hparams['batch_size']),
                shuffle=True,
                num_workers=int(hparams['num_workers'])
            )

            val_loader = DataLoader(
                DrugTargetData(
                    data=val_norm,
                    labels=val_raw[module_str].iloc[:, -1].values,
                    gene_names=val_genes
                ),
                batch_size=int(hparams['batch_size']),
                shuffle=False,
                num_workers=int(hparams['num_workers'])
            )

            test = {}
            for key, modalities in test_raw.items():
                normed = scaler.transform(modalities[module_str].iloc[:, :-1].values)
                test[key] = DataLoader(
                    DrugTargetData(
                        data=normed,
                        labels=modalities[module_str].iloc[:, -1].values,
                        gene_names=test_genes[key],
                        test_source=key
                    ),
                    batch_size=len(modalities[module_str]),
                    shuffle=False,
                    num_workers=int(hparams['num_workers'])
                )
            train_combined[module_str] = train_loader
            val_combined[module_str] = val_loader
            test_combined[module_str] = test
            label_imbalance = train_loader.dataset.label_imbalance().item()

    return train_combined, val_combined, test_combined, label_imbalance


def initialise_model(train_raw, val_raw, labels, train_genes, val_genes, test_genes, test, num_features, config):
    hyperparams = config['hyperparameters']
    train_combined, val_combined, test_combined, train_imbalance = normalise_data(train_raw, val_raw, labels,
                                                                                  train_genes, val_genes, test_genes,
                                                                                  test, config, module_str)

    max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in train_raw['pvc'].keys()])
    with open("../data/elgh/missense_mutation_map.pkl", "rb") as f:
        missense_map = pkl.load(f)
    num_mutations = len(missense_map)

    gc_features_dim = train_raw['gc'].shape[1] - 1
    go_features_dim = train_raw['go'].shape[1] - 1

    base = MultiModalTargetIdentifier(
        config=config,
        num_features_gc=gc_features_dim,
        num_features_go=go_features_dim,
        num_mutations=num_mutations,
        max_seq_len=hyperparams['max_seq_len'],
        num_genes=max_genes_pvc
    )

    model = MultiModalLightningTargetIdentifier(
        model=base,
        config=config,
        imbalance=train_imbalance
    )

    # if module_str == "pvc":
    #     varformer = ShardedVarformerTargetIdentifier(
    #         config=config,
    #         num_features=num_features,
    #         num_mutations=num_mutations,
    #         max_seq_len=hyperparams['max_seq_len'],
    #         num_genes=num_genes
    #     )
    #
    #     model = ShardedVarformerLightningTargetIdentifier(
    #         model=varformer,
    #         config=config,
    #         imbalance=train_imbalance
    #     )
    # else:
    #     model = MLPLightningTargetIdentifier(
    #         model=mlp_pytorch,
    #         config=config,
    #         imbalance=train_imbalance
    #     )

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    hyperparameters = dict(
        depth=hyperparams['num_layers'],
        lr=hyperparams['lr_start'],
        batch_size=hyperparams['batch_size'],
        optimizer=hyperparams['optimizer'],
        epochs=hyperparams['epochs'],
        dropout=hyperparams['dropout'],
        width=hyperparams['width'],
        weight_decay=hyperparams['weight_decay']
    )

    return model, _train, val, test, hyperparameters, accelerator


def train(tag="Training", module_str=None, wandb_run=None):
    if module_str == "pvc":
        mdp = ModuleDataProcessor(gc=False, go=False, pvc=True, psc=False)
    else:
        mdp = ModuleDataProcessor(gc=True, go=False, pvc=False, psc=False)

    gcp = mdp.open_gc_data()
    if module_str == "pvc":
        pvc = mdp.open_pvc_data(gc_data=gcp, tune=True)
        data = pvc.data
        num_features = gcp.num_features
        config = mdp.config
        labels = data['labels']
        data.pop('labels')
    else:
        data = gcp.data
        num_features = gcp.num_features
        config = gcp.config
        labels = gcp.labels

    test_data = None
    test_genes = None

    gene_ids = list(data.keys())

    # Split keys into train and test sets
    train_ids, val_ids = train_test_split(gene_ids, test_size=0.2, random_state=42)

    # Create train and test dictionaries
    train_raw = {gene_id: data[gene_id] for gene_id in train_ids}
    val_raw = {gene_id: data[gene_id] for gene_id in val_ids}

    train_genes = pd.Index(train_ids)
    val_genes = pd.Index(val_ids)

    mlp_lightning, _train, val, test, hyperparameters, accelerator = initialise_model(
        train_raw,
        val_raw,
        labels,
        train_genes,
        val_genes,
        test_genes,
        test_data,
        num_features,
        config,
        module_str
    )

    if tag == "Standard Training":
        if config['hyperparameters']['mlp']['wandb']:
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
        if config['hyperparameters']['wandb']:
            utils.set_seed(42)
            run = wandb_run
            lr_monitor = LearningRateMonitor(logging_interval='step')
            if torch.cuda.device_count() > 1:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[lr_monitor],
                    precision=config['hyperparameters']['precision'],
                    strategy="ddp_find_unused_parameters_true",
                    devices=-1,
                    gradient_clip_val=1,
                    deterministic=True
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[checkpoint_callback, lr_monitor],
                    gradient_clip_val=1,
                    deterministic=True
                )

            trainer.fit(mlp_lightning, _train, val)
            run.finish()

            return trainer
    else:
        raise ValueError("Invalid tag. Pick from 'Standard Training', 'PUUPL Training' or 'Tuning'")


def kfold_train(
        data: Union[pd.DataFrame, dict],
        model_type: str,
        modules: Union[str, Dict[str, bool]]
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
        os.mkdir(f'checkpoints/{group}')
    else:
        i = 1
        while os.path.isdir(f'checkpoints/{group}'):
            i += 1
            group = f"distillation-{i}-{model_type}-{module_str}"
        os.mkdir(f'checkpoints/{group}')
    
    gc_data = data['train']['gc']

    go_data = data['train']['go']

    pvc_data = data['train']['pvc']
    pvc_labels = pvc_data['labels']
    pvc_data.pop('labels')

    genes = data['genes']

    num_features = data['num_features']

    test_data = data['test_data']
    test_genes = data['test_genes']

    config = data['config']

    for fold, (train_indices, val_indices) in enumerate(kfold.split(gc_data)):
        print(f"Training fold {fold + 1}/{num_splits}")

        gc_train_raw = gc_data.iloc[train_indices, :]
        gc_val_raw = gc_data.iloc[val_indices, :]

        go_train_raw = go_data.iloc[train_indices, :]
        go_val_raw = go_data.iloc[val_indices, :]

        train_genes = [genes[i] for i in train_indices]
        val_genes = [genes[i] for i in val_indices]
        pvc_train_raw = {k: v for k, v in pvc_data.items() if k in train_genes}
        pvc_val_raw = {k: v for k, v in pvc_data.items() if k in val_genes}

        train_raw = {
            'gc': gc_train_raw,
            'go': go_train_raw,
            'pvc': pvc_train_raw
        }

        val_raw = {
            'gc': gc_val_raw,
            'go': go_val_raw,
            'pvc': pvc_val_raw
        }

        model, _train, val, test, hyperparameters, accelerator = initialise_model(
            train_raw,
            val_raw,
            pvc_labels,
            train_genes,
            val_genes,
            test_genes,
            test_data,
            num_features,
            config
        )

        if config['hyperparameters']['wandb']:
            run = wandb.init(
                project="drug-target-prediction",
                tags=[f"distillation-fold{fold + 1}"],
                config=hyperparameters,
                group=f"{group}"
            )
            run_name = wandb.run.name

            early_stopping_callback = EarlyStopping(
                monitor='val_auroc',  # Metric to monitor
                patience=100,  # Number of epochs with no improvement after which training will be stopped
                verbose=True,  # Verbosity mode
                mode='max'  # Mode can be 'min', 'max', or 'auto'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_auroc',
                dirpath=f'checkpoints/{group}',
                filename=f'{run_name}' + '-{epoch:02d}-{val_auroc:.2f}-' + f'fold{fold}',
                save_top_k=1,
                mode='max'
            )

            utils.set_seed(42)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            if torch.cuda.device_count() > 1:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[checkpoint_callback, lr_monitor],
                    precision=config['hyperparameters']['precision'],
                    strategy="ddp_find_unused_parameters_true",
                    devices=-1,
                    gradient_clip_val=1,
                    deterministic=True
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    logger=WandbLogger(wandb.run),
                    callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
                    gradient_clip_val=1,
                    deterministic=True
                )

            trainer.fit(model, _train, val)
            trainer.test(dataloaders=test["pfam"])
            trainer.test(dataloaders=test["rcnt"])
            trainer.test(dataloaders=test["pharos"])

            run.finish()

        else:
            utils.set_seed(42)
            checkpoint_callback = ModelCheckpoint(
                monitor='epoch',
                dirpath=f'checkpoints/{group}',
                save_top_k=1,
                mode='max'
            )

            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=1,
                logger=False,
                enable_checkpointing=True,
                callbacks=[checkpoint_callback]
            )
            trainer.fit(mlp_lightning, _train, val)
            trainer.test(dataloaders=test["pfam"])
            trainer.test(dataloaders=test["rcnt"])
            trainer.test(dataloaders=test["pharos"])


def kfold_teacher(**modules):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

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

    kfold_train(data, model_type="teacher", modules=modules)


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
