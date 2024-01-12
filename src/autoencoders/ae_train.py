from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.autoencoders.autoencoder import AutoencoderTrainer
from src.dataloader import VariantPathogenicityData
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import wandb


def train(data_dict, config):
    # TODO:
    #  - [X] setup training
    #  - [ ] fix autoencoding with batch size > 1

    hparams = config['hyperparameters']['pathogenicity_autoencoder']

    # wandb.init(project='danio-autoencoders')

    variant_pathogenicity = DataLoader(
        VariantPathogenicityData(data_dict=data_dict),
        batch_size=hparams['batch_size'],
        shuffle=True
    )

    model = AutoencoderTrainer(input_dim=hparams['input_dim'], output_dim=hparams['output_dim'],
                               encoding_dim=hparams['latent_dim'], num_layers=hparams['num_layers'],
                               nhead=hparams['nhead'], reduction_type=hparams['reduction'])

    # wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=30)
    trainer.fit(model, variant_pathogenicity)
