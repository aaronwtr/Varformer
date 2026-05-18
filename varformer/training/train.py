"""Train a Varformer model for one (population, seed) configuration."""
import os
import datetime
import torch
import wandb

import pytorch_lightning as pl

from varformer.utils.seeding import set_seed

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

from varformer.data.loaders import ModelPreprocessorEval
from varformer.training.callbacks import BestThresholdCallback


def train_model(data):
    """Train a Varformer model for one (population, seed) configuration."""
    config = data['config']

    # Initialize wandb run
    run = wandb.init(
        project="drug-target-prediction",
        config=config['hyperparameters'],
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

    set_seed(config['hyperparameters']['seed'])

    # Build trainer kwargs, adding conditional options only when needed.
    trainer_kwargs = dict(
        max_epochs=int(config['hyperparameters']['epochs']),
        accelerator=accelerator,
        enable_progress_bar=True,
        precision=config['hyperparameters']['precision'],
        logger=WandbLogger(wandb.run),
        callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
        gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
        deterministic=True,
    )

    if torch.cuda.device_count() > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
        trainer_kwargs["devices"] = -1
        trainer_kwargs["log_every_n_steps"] = 1
    else:
        if config['hyperparameters']['grad_accum'] is not None:
            trainer_kwargs["accumulate_grad_batches"] = config['hyperparameters']['grad_accum']
            trainer_kwargs["log_every_n_steps"] = 1
        else:
            trainer_kwargs["log_every_n_steps"] = 10

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_combined, val_combined)

    print("Starting testing...")
    trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
    trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')
    run.finish()

    best_path = checkpoint_callback.best_model_path
    return best_path if best_path else None


class VarformerTrainer:
    """Facade for training one or more Varformer models on a population.

    Wraps ``varformer.training.train.train_model`` to provide a clean SDK
    interface.  Each call to ``fit()`` trains one model per requested seed,
    returning the best checkpoint path for each.

    Prefer constructing instances through the ``Varformer.trainer()`` class
    method rather than calling ``__init__`` directly:

    Example:
        >>> trainer = Varformer.trainer("sas", config_overrides={"epochs": 50})
        >>> ckpt_paths = trainer.fit(seeds=[7, 42, 85])
        >>> print(ckpt_paths[0])
        PosixPath('checkpoints/14-05-2026/seed7-epoch=49-val_spearman=0.59.ckpt')
    """

    def __init__(self, population: str, config_overrides=None, output_dir=None):
        from varformer.config import Config
        self.population = population
        self.config = Config.load(hyperparams_override=config_overrides or {})
        self.output_dir = output_dir

    def fit(self, seeds: list) -> list:
        """Train one model per seed and return the best checkpoint path per seed.

        For each seed the full data pipeline is executed
        (``ModuleDataProcessor``), the Lightning training loop runs to
        completion, and the checkpoint with the highest ``val_spearman`` on the
        validation set is selected by ``ModelCheckpoint``.

        Args:
            seeds: List of integer random seeds to train.  Each seed produces
                an independent model run with deterministic weight
                initialisation and data shuffling.  Common practice is to pass
                ``[7, 42, 85, 123, 256]`` for a five-seed ensemble.

        Returns:
            A list of ``pathlib.Path`` objects, one per seed, pointing to the
            best checkpoint file saved during that run.  Seeds whose training
            run does not produce a checkpoint (e.g. due to an early error) are
            silently omitted, so the returned list may be shorter than
            ``seeds``.

        Example:
            >>> trainer = Varformer.trainer("nfe")
            >>> paths = trainer.fit(seeds=[42])
            >>> model = Varformer.from_checkpoint(paths[0])
        """
        from pathlib import Path
        from varformer.data.pipeline import ModuleDataProcessor

        paths: list = []
        for seed in seeds:
            cfg = {
                "hyperparameters": {
                    **self.config.hyperparameters.model_dump(),
                    "seed": int(seed),
                    "population": self.population,
                    "mode": "eval",
                },
                "paths": self.config.paths,
            }
            data = ModuleDataProcessor(
                gc=True, go=True, pvc=cfg["hyperparameters"]["use_pvc"], psc=False, config=cfg
            ).process()
            ckpt_path = train_model(data)
            if ckpt_path is not None:
                paths.append(Path(ckpt_path))
        return paths
