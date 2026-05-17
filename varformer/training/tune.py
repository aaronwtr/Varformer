"""Hyperparameter tuning with Optuna for the Varformer model.

Moved from src/training.py (tune + objective) in Phase 5.
"""
import os
import torch
import optuna
import wandb

import pytorch_lightning as pl

from varformer.utils.seeding import set_seed

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback

from varformer.data.pipeline import ModuleDataProcessor
from varformer.data.loaders import ModelPreprocessorEval


def tune():
    sampler = TPESampler()
    study = optuna.create_study(
        study_name="gdtp_varformer_tpe",
        direction="maximize",
        sampler=sampler
    )
    n_trials = 100  # Initial trial budget for TPE (adjust as needed)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial, optimized for auROC:")
    best_trial = study.best_trial

    print("  Metric: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


def objective(trial: optuna.trial.Trial) -> float:
    from varformer.config import Config
    config = Config.load()
    pl.seed_everything(config['hyperparameters']['seed'])
    # Explicit categorical values for hyperparameters
    config['hyperparameters']['lr_start'] = trial.suggest_categorical(
        'lr_start', [1e-5, 3e-5, 1e-4])  # Retain all – 3e-5 performed best overall
    config['hyperparameters']['lr_fraction'] = trial.suggest_categorical(
        'lr_fraction', [1 / 10, 1 / 100])
    config['hyperparameters']['lr_end'] = float(config['hyperparameters']['lr_start']) * float(
        config['hyperparameters']['lr_fraction'])
    config['hyperparameters']['weight_decay'] = trial.suggest_categorical(
        'weight_decay',
        [5e-4, 3e-4, 5e-3, 3e-3])
    config['hyperparameters']['batch_size'] = trial.suggest_categorical(
        'batch_size', [64, 128, 256, 512])
    config['hyperparameters']['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
    config['hyperparameters']['depth_cls_head'] = trial.suggest_categorical('depth_cls_head', [2, 4, 6])
    config['hyperparameters']['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers', [4, 6, 8])
    config['hyperparameters']['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
    config['hyperparameters']['gc_width'] = trial.suggest_categorical('gc_width', [16, 32, 64])
    config['hyperparameters']['go_width'] = trial.suggest_categorical('go_width', [256, 512])
    config['hyperparameters']['d_model'] = trial.suggest_categorical('d_model', [256, 512, 768, 1024])
    config['hyperparameters']['gv_attn_dim'] = trial.suggest_categorical('gv_attn_dim', [512, 1024, 2048])
    config['hyperparameters']['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', [1024, 2048, 4096])

    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingLR'])
    config['hyperparameters']['scheduler'] = scheduler_name

    if scheduler_name == 'CosineAnnealingLR':
        config['hyperparameters']['T0'] = trial.suggest_categorical('T0_cosine', [100, 200])
    elif scheduler_name == 'ExponentialLR':
        config['hyperparameters']['gamma'] = trial.suggest_categorical('gamma_exp', [0.85, 0.9, 0.95, 0.99])
    # No scheduler case can be handled by default in the training loop if scheduler is None

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
        weight_decay=config['hyperparameters']['weight_decay']
    )

    # Add only the scheduler-specific parameters that are relevant
    if scheduler_name == 'CosineAnnealingLR':
        hyperparameters['T0'] = config['hyperparameters']['T0']
    elif scheduler_name == 'StepLR':
        hyperparameters['step_size'] = config['hyperparameters']['step_size']
        hyperparameters['gamma'] = config['hyperparameters']['gamma']
    elif scheduler_name == 'ExponentialLR':
        hyperparameters['gamma'] = config['hyperparameters']['gamma']
    elif scheduler_name == 'ReduceLROnPlateau':
        hyperparameters['factor'] = config['hyperparameters']['factor']
        hyperparameters['patience'] = config['hyperparameters']['patience']

    # Initialize a wandb run
    run = wandb.init(
        project="varformer-hyperparameter-tuning",
        dir="/data/scratch/bty174/genomic-drug-targeting/src/",
        config=hyperparameters,
        group="varformer-tuning-run-6"
    )

    data = ModuleDataProcessor(True, True, True, False).process()

    preprocessor = ModelPreprocessorEval(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    set_seed(config['hyperparameters']['seed'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = PyTorchLightningPruningCallback(trial, monitor='val_f1')

    if torch.cuda.device_count() > 1:
        trainer_kwargs = {
            'max_epochs': int(config['hyperparameters']['epochs']),
            'accelerator': accelerator,
            'enable_progress_bar': True,
            'log_every_n_steps': 1,
            'precision': config['hyperparameters']['precision'],
            'logger': WandbLogger(wandb.run),
            'callbacks': [lr_monitor, early_stop_callback],
            'strategy': "ddp_find_unused_parameters_true",
            'devices': -1,
            'deterministic': True,
            'num_sanity_val_steps': 0
        }
    else:
        # eff_batch_size = int(config['hyperparameters']['grad_accum']) * int(config['hyperparameters']['batch_size'])
        # print(f"\nTraining with effective batch size {eff_batch_size}\n")
        if config['hyperparameters']['grad_accum'] is not None:
            trainer_kwargs = {
                'max_epochs': int(config['hyperparameters']['epochs']),
                'accelerator': accelerator,
                'enable_progress_bar': True,
                'log_every_n_steps': 1,
                'precision': config['hyperparameters']['precision'],
                'logger': WandbLogger(wandb.run),
                'callbacks': [lr_monitor, early_stop_callback],
                'deterministic': True,
                'num_sanity_val_steps': 0
            }
        else:
            trainer_kwargs = {
                'max_epochs': int(config['hyperparameters']['epochs']),
                'accelerator': accelerator,
                'enable_progress_bar': True,
                'log_every_n_steps': 1,
                'precision': config['hyperparameters']['precision'],
                'logger': WandbLogger(wandb.run),
                'callbacks': [lr_monitor, early_stop_callback],
                'deterministic': True,
                'accumulate_grad_batches': config['hyperparameters']['grad_accum'],
                'num_sanity_val_steps': 0
            }

    # Only add gradient_clip_val if it exists and is not None
    if config['hyperparameters'].get('gradient_clip_val') is not None:
        trainer_kwargs['gradient_clip_val'] = config['hyperparameters']['gradient_clip_val']

    # Create the trainer with the kwargs
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer initialized successfully")

    print("Starting fit . . . ")
    try:
        trainer.fit(model, train_combined, val_combined)
        trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')
        run.finish()
        # remove the lightning logs directory if it exists
        if os.path.exists("lightning_logs"):
            os.system("rm -r lightning_logs")
        return trainer.callback_metrics["val_f1"].item()
    except Exception as e:
        print(f"{e}\n")
        print("Out-of-memory error. Architecture is too big.")
        run.finish()
        return 0.0
