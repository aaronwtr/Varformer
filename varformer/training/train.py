"""Model training for the Varformer model (eval mode).

Moved from src/training.py (train_model) in Phase 5.
"""
import os
import datetime
import torch
import wandb
import utils

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

from varformer.data.loaders import ModelPreprocessorEval
from varformer.training.callbacks import BestThresholdCallback


def train_model(data):
    """Train the model in evaluation mode"""
    torch.set_float32_matmul_precision('medium')
    config = data['config']

    # Initialize wandb run
    hyperparameters = dict(
        lr_start=config['hyperparameters']['lr_start'],
        lr_end=config['hyperparameters']['lr_end'],
        T0=config['hyperparameters']['T0'],
        cls_depth=config['hyperparameters']['depth_cls_head'],
        num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
        nhead=config['hyperparameters']['nhead'],
        gc_width=config['hyperparameters']['gc_width'],
        go_width=config['hyperparameters']['go_width'],
        d_model=config['hyperparameters']['d_model'],
        dim_feedforward=config['hyperparameters']['dim_feedforward'],
        gv_attn_dim=config['hyperparameters']['gv_attn_dim'],
        batch_size=config['hyperparameters']['batch_size'],
        optimizer=config['hyperparameters']['optimizer'],
        epochs=config['hyperparameters']['epochs'],
        dropout=config['hyperparameters']['dropout'],
        weight_decay=config['hyperparameters']['weight_decay'],
        threshold=config['hyperparameters']['threshold']
    )

    run = wandb.init(
        project="drug-target-prediction",
        config=hyperparameters,
        group=f"varformer-{config['hyperparameters']['population']}"
    )

    preprocessor = ModelPreprocessorEval(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params})

    # Setup training callbacks
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    checkpoint_dir = f'checkpoints/{current_date}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_spearman',
        dirpath=checkpoint_dir,
        filename=f"seed{config['hyperparameters']['seed']}" + '-{epoch:02d}-{val_spearman:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    best_threshold_callback = BestThresholdCallback(
        monitor='val_spearman',
        mode='max'
    )

    utils.utils.set_seed(config['hyperparameters']['seed'])

    # Configure trainer based on available GPUs
    if torch.cuda.device_count() > 1:
        trainer = pl.Trainer(
            max_epochs=int(config['hyperparameters']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=WandbLogger(wandb.run),
            callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
            precision=config['hyperparameters']['precision'],
            strategy="ddp_find_unused_parameters_true",
            devices=-1,
            gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
            deterministic=True
        )
    else:
        if config['hyperparameters']['grad_accum'] is not None:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=config['hyperparameters']['precision'],
                logger=WandbLogger(wandb.run),
                callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
                gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                accumulate_grad_batches=config['hyperparameters']['grad_accum'],
                # limit_train_batches=0.1,  # Limit to 10% of training data for debugging purposes
                deterministic=True
            )
        else:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=10,
                precision=config['hyperparameters']['precision'],
                logger=WandbLogger(wandb.run),
                callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
                gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                deterministic=True
            )

    trainer.fit(model, train_combined, val_combined)

    print("Starting testing...")
    trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')
    run.finish()
