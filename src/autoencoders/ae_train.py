from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.autoencoders.autoencoder import AutoencoderTrainer
from src.dataloader import VariantPathogenicityData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import torch


def train(data_dict, config):
    hparams = config['hyperparameters']['pathogenicity_autoencoder']

    wandb.init(project='danio-autoencoders')

    variant_pathogenicity = DataLoader(
        VariantPathogenicityData(data_dict=data_dict, reduct_dim=hparams['input_dim'],
                                 reduction_type=hparams['reduction']),
        batch_size=hparams['batch_size'],
        collate_fn=padding,
        shuffle=True
    )

    model = AutoencoderTrainer(input_dim=hparams['input_dim'], output_dim=hparams['output_dim'],
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


def padding(batch):
    X = []
    for x, reduct_dim in batch:
        current_dimension = x.size(-1)
        if current_dimension == reduct_dim:
            X.append(x)
        elif current_dimension < reduct_dim:
            padding_left = (reduct_dim - current_dimension) // 2
            padding_right = reduct_dim - current_dimension - padding_left
            padded_x = F.pad(x, (padding_left, padding_right), value=0)
            X.append(padded_x)
        else:
            pooled_x = pooling(x, reduct_dim)
            pooled_x = pooled_x.squeeze(0)
            X.append(pooled_x)
    X = torch.stack(X)
    return X


def pooling(x, reduct_dim):
    x = x.unsqueeze(0)
    pool = nn.AdaptiveAvgPool1d(reduct_dim)
    pooled_x = pool(x)
    return pooled_x
