import gc

import yaml
import os
import torch
import optuna
import wandb
import datetime

import utils

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import pickle as pkl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, Optional
from tqdm import tqdm
from optuna.samplers import GridSampler

from dataloader import DrugTargetData, ModuleDataProcessor, VarformerDataset, MultiModalData, MultiModalDataLoader
from models.target_identifier import MultiModalTargetIdentifier
from models.lightning import (MLPLightningTargetIdentifier, ShardedVarformerLightningTargetIdentifier,
                              MultiModalLightningTargetIdentifier)
from preprocessing import ModelPreprocessor


def tune(grid=True):
    if grid:
        # todo: clean up below (make dynamic)
        search_space = {
            'threshold': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        }

        sampler = GridSampler(search_space)
        study = optuna.create_study(
            study_name="gdtp_varformer",
            direction="maximize",
            sampler=sampler
        )

        n_trials = (len(search_space['threshold']))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial, optimized for auROC:")
        best_trial = study.best_trial

        print("  Metric: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))
    else:
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
        # TODO: clean up below
        # lr_start = trial.suggest_categorical('lr_start', [1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        # lr_end = trial.suggest_categorical('lr_end', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        #
        # if lr_end <= lr_start:
        #     raise optuna.TrialPruned(f"Invalid combination: lr_end ({lr_end}) <= lr_start ({lr_start})")
        #
        # # Code here will not run if the trial is pruned
        # config['hyperparameters']['lr_start'] = lr_start
        # config['hyperparameters']['lr_end'] = lr_end
        # config['hyperparameters']['T_0'] = trial.suggest_categorical('T0', [50, 100, 200, 400, 1000])
        # config['hyperparameters']['weight_decay'] = trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.01, 0.1])

        # config['hyperparameters']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 4, 8])
        # config['hyperparameters']['nhead'] = trial.suggest_categorical('nhead', [1, 2, 4, 8])
        # config['hyperparameters']['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers', [1, 2, 4, 8])
        # config['hyperparameters']['width'] = trial.suggest_categorical('width', [64, 256, 768, 2048, 4096])
        # config['hyperparameters']['d_model'] = trial.suggest_categorical('d_model', [64, 256, 768, 2048, 4096])

        config['hyperparameters']['threshold'] = trial.suggest_float('threshold', 0.1, 0.9, step=0.05)

        config_hpo = config

        hyperparameters = dict(
            lr_start=config['hyperparameters']['lr_start'],
            lr_end=config['hyperparameters']['lr_end'],
            T0=config['hyperparameters']['T0'],
            depth=config['hyperparameters']['num_layers'],
            num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
            nhead=config['hyperparameters']['nhead'],
            width=config['hyperparameters']['width'],
            d_model=config['hyperparameters']['d_model'],
            batch_size=config['hyperparameters']['batch_size'],
            optimizer=config['hyperparameters']['optimizer'],
            epochs=config['hyperparameters']['epochs'],
            dropout=config['hyperparameters']['dropout'],
            weight_decay=config['hyperparameters']['weight_decay'],
            threshold=config['hyperparameters']['threshold']
        )

        # Initialize a wandb run
        run = wandb.init(
            project="varformer-hyperparameter-tuning",
            dir="/data/scratch/bty174/genomic-drug-targeting/src/",
            config=hyperparameters
        )

        data = ModuleDataProcessor(True, True, True, False).process()

        gc_data = data['train']['gc']

        go_data = data['train']['go']

        pvc_data = data['train']['pvc']
        pvc_data.pop('labels')

        labels = gc_data['target'].to_dict()
        test_labels = data["test_labels"]

        # remove the target column from gc_data and go_data
        gc_data = gc_data.drop(columns=['target'])
        go_data = go_data.drop(columns=['target'])

        pvc_ensgs = set(list(pvc_data.keys()))
        gc_ensgs = set(gc_data.index)

        ensgs_not_in_pvc = pvc_ensgs - gc_ensgs

        # filter out the rows from gc dataframe based on the index that are not in pvc
        # gc_data = gc_data[~gc_data.index.isin(ensgs_not_in_pvc)]
        # go_data = go_data[~go_data.index.isin(ensgs_not_in_pvc)]

        genes = data['genes']

        num_features = data['num_features']

        test_data = data['test_data']
        test_genes = data['test_genes']

        config = data['config']

        # Split keys into train and test sets
        train_genes, val_genes = train_test_split(genes, test_size=0.2, random_state=42)

        gc_train_raw = gc_data.loc[train_genes, :]
        gc_val_raw = gc_data.loc[val_genes, :]

        go_train_raw = go_data.loc[train_genes, :]
        go_val_raw = go_data.loc[val_genes, :]

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

        model, train_combined, val_combined, test_combined, hyperparameters, accelerator = initialise_model(
            train_raw,
            val_raw,
            labels,
            test_labels,
            train_genes,
            val_genes,
            test_genes,
            test_data,
            num_features,
            config_hpo
        )

        model_summary = ModelSummary(model)
        total_params = model_summary.total_parameters

        wandb.config.update({"total_params": total_params})

        if config['hyperparameters']['wandb']:
            utils.utils.set_seed(42)
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
                    gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                    deterministic=True
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=int(config['hyperparameters']['epochs']),
                    accelerator=accelerator,
                    enable_progress_bar=True,
                    log_every_n_steps=1,
                    precision=config['hyperparameters']['precision'],
                    logger=WandbLogger(wandb.run),
                    callbacks=[lr_monitor, checkpoint_callback],
                    gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                    deterministic=True
                )
        else:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=config['hyperparameters']['precision'],
                logger=False,
                enable_checkpointing=True,
                gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                callbacks=[checkpoint_callback]
            )

        try:
            trainer.fit(model, train_combined, val_combined)
            run.finish()
            return trainer.callback_metrics["val_auroc"].item()
        except Exception as e:
            print(f"{e}\n")
            print("Out-of-memory error. Architecture is too big.")
            run.finish()
            return 0.0


def normalise_data_archive(train_raw, val_raw, labels, train_genes, val_genes, test_genes, test_raw, config):
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


def normalise_data(train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test_raw, torch_dtype,
                   config):
    hparams = config['hyperparameters']

    # Initialize dictionaries to store datasets for each split
    train_datasets = {}
    val_datasets = {}
    test_datasets = {key: {} for key in test_raw.keys()}
    scalers = {}

    # First create all datasets
    for module_str, train_data in train_raw.items():
        if module_str != "pvc":
            # Handle non-PVC modalities
            val_norm = val_raw[module_str].values
            train_norm = train_data.values

            scaler = MinMaxScaler()
            train_norm = scaler.fit_transform(train_norm)
            val_norm = scaler.transform(val_norm)
            scalers[module_str] = scaler

            train_norm = {gene: train_norm[i] for i, gene in enumerate(train_genes)}
            val_norm = {gene: val_norm[i] for i, gene in enumerate(val_genes)}

            train_datasets[module_str] = MultiModalData(
                data=train_norm,
                labels=labels,
                gene_names=train_genes,
                dtype=torch_dtype
            )

            val_datasets[module_str] = MultiModalData(
                data=val_norm,
                labels=labels,
                gene_names=val_genes,
                dtype=torch_dtype
            )

            # Create test datasets for each test source
            for key, modalities in test_raw.items():
                normed = scaler.transform(modalities[module_str].values)
                normed = {gene: normed[i] for i, gene in enumerate(test_genes[key][module_str])}
                test_datasets[key][module_str] = MultiModalData(
                    data=normed,
                    labels=test_labels,
                    gene_names=test_genes[key][module_str],
                    dtype=torch_dtype,
                    test_source=key
                )
        else:
            # Handle PVC modality
            train_datasets[module_str] = MultiModalData(
                data=None,
                labels=None,
                gene_names=train_genes,
                dtype=torch_dtype,
                variant_data={'data': train_data, 'labels': labels},
                max_variants=hparams['max_seq_len']
            )

            val_datasets[module_str] = MultiModalData(
                data=None,
                labels=None,
                gene_names=val_genes,
                dtype=torch_dtype,
                variant_data={'data': val_raw[module_str], 'labels': labels},
                max_variants=hparams['max_seq_len']
            )

            # Create test datasets for each test source
            for key, modalities in test_raw.items():
                test_datasets[key][module_str] = MultiModalData(
                    data=None,
                    labels=None,
                    gene_names=test_genes[key][module_str],
                    dtype=torch_dtype,
                    variant_data={
                        'data': modalities[module_str],
                        'labels': test_labels,
                        'test_source': key
                    },
                    max_variants=hparams['max_seq_len'],
                    test_source=key
                )

    # Create synchronized dataloaders
    train_loader = MultiModalDataLoader(
        datasets=train_datasets,
        batch_size=hparams['batch_size'],
        shuffle=True
    )

    val_loader = MultiModalDataLoader(
        datasets=val_datasets,
        batch_size=hparams['batch_size'],
        shuffle=False
    )

    # Create test loaders for each test source
    test_loaders = {}
    for key in test_raw.keys():
        test_loaders[key] = MultiModalDataLoader(
            datasets=test_datasets[key],
            batch_size=len(next(iter(test_datasets[key].values()))),  # Use length of any modality
            shuffle=False
        )

    # Calculate label imbalance using any modality (they should all be the same now)
    label_imb_raw = next(iter(train_datasets.values())).label_imbalance()
    label_imbalance = label_imb_raw.item() if isinstance(label_imb_raw, torch.Tensor) else label_imb_raw

    return train_loader, val_loader, test_loaders, label_imbalance


def initialise_model(train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test, num_features,
                     torch_dtype, config):
    hyperparams = config['hyperparameters']
    train_combined, val_combined, test_combined, train_imbalance = normalise_data(train_raw, val_raw, labels,
                                                                                  test_labels, train_genes, val_genes,
                                                                                  test_genes, test, torch_dtype, config)

    max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in train_raw['pvc'].keys()])
    with open("../data/elgh/missense_mutation_map.pkl", "rb") as f:
        missense_map = pkl.load(f)
    num_mutations = len(missense_map)

    gc_features_dim = train_raw['gc'].shape[1]
    go_features_dim = train_raw['go'].shape[1]

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
        imbalance=train_imbalance,
        num_iters=len(train_combined)
    )

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

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

    return model, train_combined, val_combined, test_combined, hyperparameters, accelerator


def train_model(data):
    torch.set_float32_matmul_precision('medium')
    config = data['config']

    # Initialize wandb run
    hyperparameters = dict(
        lr_start=config['hyperparameters']['lr_start'],
        lr_end=config['hyperparameters']['lr_end'],
        T0=config['hyperparameters']['T0'],
        depth=config['hyperparameters']['num_layers'],
        num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
        nhead=config['hyperparameters']['nhead'],
        width=config['hyperparameters']['width'],
        d_model=config['hyperparameters']['d_model'],
        batch_size=config['hyperparameters']['batch_size'],
        optimizer=config['hyperparameters']['optimizer'],
        epochs=config['hyperparameters']['epochs'],
        dropout=config['hyperparameters']['dropout'],
        weight_decay=config['hyperparameters']['weight_decay'],
        threshold=config['hyperparameters']['threshold']
    )

    run = wandb.init(
        project="drug-target-prediction",
        config=hyperparameters,
        group="multimodal-training-run-1"
    )

    preprocessor = ModelPreprocessor(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params})

    # Setup training callbacks
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    checkpoint_dir = f'checkpoints/{current_date}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        dirpath=checkpoint_dir,
        filename=f"seed{config['hyperparameters']['seed']}" + '-{epoch:02d}-{val_auroc:.2f}',
        save_top_k=1,
        mode='max'
    )

    # Configure trainer based on available GPUs
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
            gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
            deterministic=True
        )
    else:
        trainer = pl.Trainer(
            max_epochs=int(config['hyperparameters']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            precision=config['hyperparameters']['precision'],
            logger=WandbLogger(wandb.run),
            callbacks=[lr_monitor, checkpoint_callback],
            gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
            deterministic=True
        )

    trainer.fit(model, train_combined, val_combined)

    # Test on different datasets
    trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')

    run.finish()


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

    train_model(data)


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
