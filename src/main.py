import os

import yaml
from sklearn.model_selection import train_test_split
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
    labels = data.iloc[:, -1].values
    num_features = features.shape[1]
    num_classes = len(set(labels))

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train = DataLoader(
        DrugTargetData(data=train),
        batch_size=int(config['mlp']['batch_size']),
        shuffle=True
        )

    test = DataLoader(DrugTargetData(data=test),
                      batch_size=int(config['mlp']['batch_size']),
                      shuffle=False
                      )

    mlp_pytorch = PyTorchMLP(config=config, num_features=num_features, num_classes=num_classes)
    mlp_lightning = LightningMLP(model=mlp_pytorch, config=config)

    wandb_logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(
        max_epochs=int(config['mlp']['epochs']),
        accelerator='cpu',
        logger=wandb_logger,
        show_progress_bar=False
        )

    trainer.fit(mlp_lightning, train, test)

    # TODO:
    #  - Normalize features after train, test, splitting
    #  - Add early stopping
    #  - Come up with validation strategy (ACMG gene set)
    #  - Hyperparameter tuning


if __name__ == "__main__":
    main()
    