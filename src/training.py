import yaml
import os
import torch
import optuna
import wandb

import utils
import plot

import lightning as pl
import pandas as pd
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from preprocessing import GeneCharacterisationPreprocessor, VariantAndStructurePreprocessor
from dataloader import DrugTargetData, ModuleDataProcessor
from model import BaseTargetIdentifier, BaseLightningTargetIdentifier
from puupl import training as puupl_training
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union


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


def normalise_data(train_raw, val_raw, config, model_type="mlp"):
    if val_raw is not None:
        gene_names_test = val_raw.iloc[:, 0].values
        val_norm = val_raw.iloc[:, 1:8].values
        val_bin = val_raw.iloc[:, 8:-1].values

    gene_names_train = train_raw.iloc[:, 0].values
    train_norm = train_raw.iloc[:, 1:8].values

    train_bin = train_raw.iloc[:, 8:-1].values

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_norm)
    _train = np.concatenate((train_norm, train_bin), axis=1)

    _train = DataLoader(
        DrugTargetData(data=_train, labels=train_raw.iloc[:, -1].values, gene_names=gene_names_train),
        batch_size=int(config[model_type]['batch_size']),
        shuffle=True,
        num_workers=int(config[model_type]['num_workers'])
    )

    if val_raw is not None:
        val_norm = scaler.transform(val_norm)
        val = np.concatenate((val_norm, val_bin), axis=1)
        val = DataLoader(
            DrugTargetData(data=val, labels=val_raw.iloc[:, -1].values, gene_names=gene_names_test),
            batch_size=int(config['puupl']['batch_size']),
            shuffle=False
        )
        return _train, val
    return _train


def initialise_model(train_raw, val_raw, num_features, config):
    _train, val = normalise_data(train_raw, val_raw, config)

    train_imbalance = 1 / float(_train.dataset.label_imbalance().item())  # calculate inverse class frequency

    mlp_pytorch = BaseTargetIdentifier(config=config, num_features=num_features)
    mlp_lightning = BaseLightningTargetIdentifier(model=mlp_pytorch, config=config, imbalance=train_imbalance)

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    hyperparameters = dict(
        depth=config['mlp']['depth'],
        lr=config['mlp']['lr_start'],
        batch_size=config['mlp']['batch_size'],
        optimizer=config['mlp']['optimizer'],
        epochs=config['mlp']['epochs'],
        dropout=config['mlp']['dropout'],
        width=config['mlp']['width'],
        weight_decay=config['mlp']['weight_decay'],
    )

    return mlp_lightning, _train, val, hyperparameters, accelerator


def train(tag="Training"):
    gcp = ModuleDataProcessor.open_gc_data
    data = gcp.data
    num_features = gcp.num_features
    config = gcp.config

    test_genes = gcp.acmg_genes

    # Note: Validation samples are clinically actionable genes as defined by the ACMG (see:
    # https://www.coriell.org/1/NIGMS/Collections/ACMG-73-Genes). We also have a broader set of clinically actionable
    # genes - with scores - from ClinGen. These are currently loaded in `get_actionable_genes` in `preprocessing.py`.
    # These need NOT yet have FDA-approved drugs.
    test = data[data['ENSG'].isin(test_genes)]

    train_raw, val_raw = train_test_split(data, test_size=0.2, random_state=42)

    if tag == "Standard Training" or tag == "Tuning":
        mlp_lightning, train, val, hyperparameters, accelerator = initialise_model(train_raw, val_raw, num_features,
                                                                                   config)
    elif tag == "PUUPL Training":
        train, val = normalise_data(train_raw, val_raw, config, model_type="puupl")
    else:
        raise ValueError("Invalid tag. Pick from 'Standard Training', 'PUUPL Training' or 'Tuning'")

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
        trainer.fit(mlp_lightning, train, val)

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


def kfold_train(data: pd.DataFrame, num_features: int, config: dict, model_type: str,
                modules: Union[str, Dict[str, bool]]):
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
            group = f"distillation-{i}-{model_type}"
        os.mkdir(f'checkpoints/{group}')

    # use the new train object to train a new model
    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{num_splits}")

        # Split the data
        train_raw = data.iloc[train_indices, :]
        val_raw = data.iloc[val_indices, :]

        mlp_lightning, _train, val, hyperparameters, accelerator = initialise_model(train_raw, val_raw, num_features,
                                                                                   config)

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
            mode='max',
        )

        utils.set_seed(42)
        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=WandbLogger(wandb.run),
            callbacks=[checkpoint_callback]
        )

        trainer.fit(mlp_lightning, _train, val)

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
            train_df = preprocessor.data
            num_features = preprocessor.num_features
            config = preprocessor.config
            kfold_train(train_df, num_features, config, model_type="teacher", modules=module)
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
