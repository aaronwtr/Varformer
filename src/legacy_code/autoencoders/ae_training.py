import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.autoencoders.ae import AutoencoderTrainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import src.dataloader as dl

from src.utils import padding


def train(data_dict, config):
    hparams = config['hyperparameters']['pathogenicity_autoencoder']

    wandb.init(project='danio-autoencoders')

    variant_pathogenicity = DataLoader(
        dl.VariantPathogenicityData(data_dict=data_dict, reduct_dim=hparams['input_dim'],
                                 reduction_type=hparams['reduction']),
        batch_size=hparams['batch_size'],
        collate_fn=padding,
        shuffle=True
    )

    model = AutoencoderTrainer(input_dim=hparams['io_dim'],
                               encoding_dim=hparams['latent_dim'], num_layers=hparams['num_layers'],
                               nhead=hparams['nhead'], reduction_type=hparams['reduction'])

    wandb_logger = WandbLogger()

    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        dirpath='autoencoders/checkpoints/variant_pathogenicity_encoder',
        filename=f'{run_name}' + '-{epoch:02d}-{val_auroc:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(max_epochs=hparams['max_epochs'], logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, variant_pathogenicity)
