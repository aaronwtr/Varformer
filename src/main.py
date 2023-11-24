import yaml
import os
import torch
import optuna

import lightning as pl

from pytorch_lightning.loggers import WandbLogger
from preprocessing import GeneCharacterisationPreprocessor, MissenseVariantPreprocessor
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

    print("Best trial, optimized for Spearman:")
    best_trial = study.best_trial

    print("  Metric: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    with open("best_hyperparameters.txt", "a") as f:
        if os.stat("best_hyperparameters.txt").st_size == 0:
            f.write("Run ID\tSpearman Corr.\tHyperparameters\n")
            run_id = 0
        else:
            with open("best_hyperparameters.txt", "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                run_id = int(last_line.split("\t")[0]) + 1
        f.write(f"{run_id}\t{best_trial.value}\t{best_trial.params}\n")


def objective(trial: optuna.trial.Trial) -> float:
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    config = config['hyperparameters']

    config['mlp']['width_1'] = trial.suggest_categorical('width_1', [2, 6, 8, 16, 32, 64, 128])
    config['mlp']['lr'] = trial.suggest_categorical('lr', [1e-2, 1e-3, 1e-4, 1e-5])
    config['mlp']['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    config['mlp']['optimizer'] = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop',
                                                                         'SGD'])

    trainer = training(tag="Tuning")

    hyperparameters = dict(
        width_1=config['mlp']['width_1'],
        lr=config['mlp']['lr']
    )

    trainer.logger.log_hyperparams(hyperparameters)

    return trainer.callback_metrics["val_spearman"].item()


def training(tag="Training"):
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    if not os.path.exists('../data/features/pathogenicity_features.pkl'):
        MissenseVariantPreprocessor(config)

    gcp = GeneCharacterisationPreprocessor(config=config)
    print("Gene characterisation features preprocessed!\n")

    config = config['hyperparameters']

    data = gcp.data
    features = data.iloc[:, 1:-1].values
    num_features = features.shape[1]

    train_raw, test_raw = train_test_split(data, test_size=0.2, random_state=42)
    gene_names_train = train_raw.iloc[:, 0].values
    gene_names_test = test_raw.iloc[:, 0].values

    train = train_raw.iloc[:, 1:-1].values
    test = test_raw.iloc[:, 1:-1].values

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    train = DataLoader(
        DrugTargetData(data=train, labels=train_raw.iloc[:, -1].values, gene_names=gene_names_train,
                       features=list(features)),
        batch_size=int(config['mlp']['batch_size']),
        shuffle=True
    )

    test = DataLoader(
        DrugTargetData(data=test, labels=test_raw.iloc[:, -1].values, gene_names=gene_names_test,
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

    if tag == "Training":
        wandb_logger = WandbLogger(
            project="drug-target-prediction",
            tags=["mlp", {tag}],
            log_model="all"
        )

        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=False,
            logger=wandb_logger
        )
    elif tag == "Tuning":
        trainer = pl.Trainer(
            max_epochs=int(config['mlp']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=False
        )
    else:
        raise ValueError("Invalid tag. Pick from 'Training' or 'Tuning'")

    trainer.fit(mlp_lightning, train, test)

    return trainer


def main(mode="training"):
    if mode == "training":
        training()
    elif mode == "tuning":
        tuning()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning'")


if __name__ == "__main__":
    main(mode="tuning")

    # TODO:
    #  - Hyperparameter tuning
    #  - Scale up to find loss convergence epoch
    #  - Set up checkpointing to save the model with the best validation auroc
    #  - Come up with validation strategy (ACMG gene set)
