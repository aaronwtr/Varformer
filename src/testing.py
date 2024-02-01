import torch
import yaml

from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import GeneCharacterisationPreprocessor
from model import LightningMLP


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


def load_test_data(config):
    gcp = GeneCharacterisationPreprocessor(config=config)
    acmg_data = gcp.acmg_genes
    pfam_data = gcp.pfam_genes
    return acmg_data, pfam_data


def create_dataloader(acmg_data, pfam_data, batch_size=64):
    acmg_dataset = TensorDataset(torch.tensor(acmg_data, dtype=torch.float32))
    acmg_dataloader = DataLoader(acmg_dataset, batch_size=batch_size)
    pfam_dataset = TensorDataset(torch.tensor(pfam_data, dtype=torch.float32))
    pfam_dataloader = DataLoader(pfam_dataset, batch_size=batch_size)
    return acmg_dataloader, pfam_dataloader


def test_model_on_acmg_data(model, acmg_dataloader, pfam_dataloader):
    trainer = Trainer()

    # Testing on ACMG data
    trainer.test(model, dataloaders=acmg_dataloader)

    # Testing on pfam data
    trainer.test(model, dataloaders=pfam_dataloader)


def testing():
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    model_path = 'path_to_your_model.pt'  # Replace with your model path

    model = load_model(model_path)
    acmg_data, pfam_data = load_test_data(config)
    acmg_dataloader, pfam_dataloader = create_dataloader(acmg_data, pfam_data)

    test_model_on_acmg_data(model, acmg_dataloader, pfam_dataloader)
