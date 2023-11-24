import os

import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lightning as pl
from pytorch_lightning.loggers import WandbLogger

from preprocessing import GeneCharacterisationPreprocessor, MissenseVariantPreprocessor
from dataloader import DrugTargetData
from model import PyTorchMLP, LightningMLP
from torch.utils.data import DataLoader


def main():
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    if not os.path.exists('../data/features/pathogenicity_features.pkl'):
        MissenseVariantPreprocessor(config)

    gcp = GeneCharacterisationPreprocessor(config=config)
    print("Gene characterisation features preprocessed!\n")

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

    train_imbalance = 1 / float(train.dataset.label_imbalance().item())

    mlp_pytorch = PyTorchMLP(config=config, num_features=num_features)
    mlp_lightning = LightningMLP(model=mlp_pytorch, config=config)

    wandb_logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(
        max_epochs=int(config['mlp']['epochs']),
        accelerator='cpu',
        enable_progress_bar=False,
        logger=wandb_logger
        )

    trainer.fit(mlp_lightning, train, test)

    # TODO:
    #  - Normalize features after train, test, splitting
    #  - Add early stopping
    #  - Come up with validation strategy (ACMG gene set)
    #  - Hyperparameter tuning


if __name__ == "__main__":
    main()
    