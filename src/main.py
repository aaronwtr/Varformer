import yaml
from sklearn.model_selection import train_test_split
import lightning as pl
import utils

from preprocessing import GeneCharacterisationPreprocessor, MissenseVariantPreprocessor
from dataloader import DrugTargetData
from model import PyTorchMLP, LightningMLP
from torch.utils.data import DataLoader


def main():
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

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

    trainer = pl.Trainer(
        max_epochs=int(config['mlp']['epochs']),
        accelerator='cpu'
        )

    trainer.fit(mlp_lightning, train, test)

    # TODO:
    #  Add metrics and logging of metrics
    #  Plot training and validation loss
    #  Add early stopping
    #  Come up with validation strategy (ACMG gene set)
    #  Hyperparameter tuning
    #  Optional: debug mps accelerator (GPU)


if __name__ == "__main__":
    # main()
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    mvp = MissenseVariantPreprocessor(config=config, evaluation=True)
    data = mvp.variant_data
    utils.evaluate_am(data)

    # TODO: plot AUC, MCC and class weighted F1 for VP, VPGH, and AM. Use the crossval results in output folder
