from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from autoencoder import AutoencoderTrainer
from torch.utils.data import DataLoader

import wandb


def ae_train(dataset, hparams):
    wandb.init(project='danio-autoencoders')

    # TODO:
    #  - [ ] split in train and val set
    #  - [ ] when training successful, setup optuna for hyperparameter tuning

    dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True)

    model = AutoencoderTrainer(input_dim=hparams['input_dim'], encoding_dim=hparams['latent_dim'],
                               num_layers=hparams['num_layers'], nhead=hparams['nhead'], reduction=hparams['reduction'])

    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=100, logger=wandb_logger)
    trainer.fit(model, dataloader)
