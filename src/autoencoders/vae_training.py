import wandb
import torch

import numpy as np
import src.dataloader as dl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from src.autoencoders.vae import VAETrainer
from src.utils import padding


def train_vae(data_dict, config):
    hparams = config['hyperparameters']['pathogenicity_autoencoder']  # Adjust if needed for VAE config

    wandb.init(project='danio-autoencoders')

    variant_pathogenicity = DataLoader(
        dl.VariantPathogenicityData(data_dict=data_dict, reduct_dim=hparams['io_dim'],
                                    reduction_type=hparams['reduction']),
        batch_size=hparams['batch_size'],
        collate_fn=padding,
        shuffle=True
    )

    model = VAETrainer(input_dim=hparams['io_dim'], latent_dim=hparams['latent_dim'])

    wandb_logger = WandbLogger()

    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',  # May consider 'val_loss' for VAEs
        dirpath='autoencoders/checkpoints/variant_pathogenicity_vae',
        filename=f'{run_name}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(max_epochs=hparams['max_epochs'], logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, variant_pathogenicity)

    return model


def extract_latent_features(vae_model, data_loader):
    vae_model.eval()
    latent_features = []

    for batch in data_loader:
        with torch.no_grad():
            z = vae_model.predict_step(batch, 0)
            latent_features.append(z.cpu().numpy())

    return np.concatenate(latent_features, axis=0)
