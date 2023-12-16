import yaml
import os
import torch
import optuna
import utils

import lightning as pl
import pandas as pd
import numpy as np


from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from preprocessing import GeneCharacterisationPreprocessor
from dataloader import DrugTargetData
from model import PyTorchMLP, LightningMLP
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def tuning():
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
            f.write("Run ID\tauROC\tHyperparameters\n")
        run_id = 0
    else:
        with open("best_hyperparameters.txt", "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            run_id = int(last_line.split("\t")[0]) + 1

    with open("best_hyperparameters.txt", "a") as f:
        f.write(f"{run_id}\t{best_trial.value}\t{best_trial.params}\n")


def objective(trial: optuna.trial.Trial) -> float:
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    config = config['hyperparameters']

    config['mlp']['depth'] = trial.suggest_int('depth', 1, 8, step=1)
    config['mlp']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    config['mlp']['threshold'] = trial.suggest_float('threshold', 0.2, 0.8, step=0.05)

    trainer = training(tag="Tuning")

    # hyperparameters = dict(
    #     depth=config['mlp']['depth'],
    #     lr=config['mlp']['lr_start'],
    #     batch_size=config['mlp']['batch_size'],
    #     optimizer=config['mlp']['optimizer'],
    #     epochs=config['mlp']['epochs'],
    #     dropout=config['mlp']['dropout'],
    #     width=config['mlp']['width'],
    #     threshold=config['mlp']['threshold']
    # )
    #
    # trainer.logger.log_hyperparams(hyperparameters)

    return trainer.callback_metrics["val_auroc"].item()


def training(tag="Training"):
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    gcp = GeneCharacterisationPreprocessor(config=config)
    print("Gene characterisation features preprocessed!\n")

    config = config['hyperparameters']

    data = gcp.data

    val_genes = gcp.acmg_genes

    # Note: Validation samples are clinically actionable genes as defined by the ACMG (see:
    # https://www.coriell.org/1/NIGMS/Collections/ACMG-73-Genes). We also have a broader set of clinically actionable
    # genes - with scores - from ClinGen. These are currently loaded in `get_actionable_genes` in `preprocessing.py`.
    # These need NOT yet have FDA-approved drugs.
    val = data[data['ENSG'].isin(val_genes)]

    features = data.iloc[:, 1:-1].values
    num_features = features.shape[1]

    train_raw, val_raw = train_test_split(data, test_size=0.2, random_state=42)
    gene_names_train = train_raw.iloc[:, 0].values
    gene_names_test = val_raw.iloc[:, 0].values

    train_norm = train_raw.iloc[:, 1:8].values
    val_norm = val_raw.iloc[:, 1:8].values

    train_bin = train_raw.iloc[:, 8:].values
    val_bin = val_raw.iloc[:, 8:].values

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_norm)
    val_norm = scaler.transform(val_norm)

    train = np.concatenate((train_norm, train_bin), axis=1)
    val = np.concatenate((val_norm, val_bin), axis=1)

    train = DataLoader(
        DrugTargetData(data=train, labels=train_raw.iloc[:, -1].values, gene_names=gene_names_train,
                       features=list(features)),
        batch_size=int(config['mlp']['batch_size']),
        shuffle=True,
        num_workers=int(config['mlp']['num_workers'])
    )

    val = DataLoader(
        DrugTargetData(data=val, labels=val_raw.iloc[:, -1].values, gene_names=gene_names_test,
                       features=list(features)),
        batch_size=int(config['mlp']['batch_size']),
        shuffle=False
    )

    train_imbalance = 1 / float(train.dataset.label_imbalance().item())  # calculate inverse class frequency

    mlp_pytorch = PyTorchMLP(config=config, num_features=num_features)
    mlp_lightning = LightningMLP(model=mlp_pytorch, config=config, imbalance=train_imbalance)

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
        width=config['mlp']['width']
    )

    if tag == "Training":
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
    elif tag == "Tuning":
        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False
        )
    else:
        raise ValueError("Invalid tag. Pick from 'Training' or 'Tuning'")

    trainer.fit(mlp_lightning, train, val)

    return trainer


def main(mode="training"):
    if mode == "training":
        training()
    elif mode == "tuning":
        tuning()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning'")


if __name__ == "__main__":
    main(mode="training")

    # TODO:
    #  MLP model
    #  [X] Set up checkpointing to save the model with the best validation auroc
    #  [X] Make sure to save hyperparameters per model in wandb or locally
    #  [X] Hyperparameter tuning
    #  [X] Find out how many epochs to train for (500)
    #  [X] Add gene essentiality feature
    #  [X] Add biological process and molecular function features from HPA (think about implications)
    #  [X] Add cellular target localization features from HPA
    #  [X] Check normalization after introduction of new features
    #  [X] Hold-out test set (ACMG + randomly sampled negatives)
    #  [ ] Slot features into the feature types for spider plots
    #  [ ] Set up cross validation
    #  [ ] Train baseline neural network model
    #  [ ] Introduce self-destillation / self-supervision (see SSL cookbook)
    #  [ ] Add autoencoder (for the variant-level and categorical features)
    #  [ ] Train final models
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
    #  -
    #  XGBoost baseline:
    #  [ ] Set up training loops
    #  [ ] Hyperparameter tuning
    #  [ ] Set up cross validation
    #  [ ] Make training data different degrees of class imbalance and evaluate. Keep val data as is
    #  [ ] Train final model
    #  [ ] Add SHAP interpretability
