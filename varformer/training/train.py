"""Train a Varformer model for one (population, seed) configuration."""
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import torch
import wandb

import pytorch_lightning as pl

from varformer.utils.seeding import set_seed

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

from varformer.data.loaders import ModelPreprocessorEval
from varformer.training.callbacks import BestThresholdCallback, NaNDiagnosticsCallback


def _clock_now() -> str:
    """Query the runtime-host clock for immediate, single-use recording."""
    completed = subprocess.run(
        ["date", "+%Y-%m-%dT%H:%M:%S%z"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{4}",
        value,
    ):
        raise ValueError(f"Runtime-host date returned an unexpected timestamp: {value!r}")
    return value


def _clock_date_directory() -> str:
    """Query the runtime-host date for a legacy date-named checkpoint directory."""
    completed = subprocess.run(
        ["date", "+%d-%m-%Y"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not re.fullmatch(r"[0-9]{2}-[0-9]{2}-[0-9]{4}", value):
        raise ValueError(f"Runtime-host date returned an unexpected directory date: {value!r}")
    return value


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _configuration_hash(configuration: dict) -> str:
    encoded = json.dumps(_jsonable(configuration), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _gene_list_hash(genes) -> str:
    """Order-independent SHA-256 for a gene set."""
    return hashlib.sha256(("\n".join(sorted(map(str, genes))) + "\n").encode()).hexdigest()


def _split_manifest_payload(preprocessor, population: str, seed: int) -> dict:
    """Build and validate the exact gene-level train/validation/test split."""
    train_genes = list(map(str, preprocessor.train_genes))
    validation_genes = list(map(str, preprocessor.val_genes))
    tests = {
        str(task): list(map(str, genes))
        for task, genes in preprocessor.test_labels_per_source.items()
    }
    train_set = set(train_genes)
    validation_set = set(validation_genes)
    test_union = set().union(*(set(genes) for genes in tests.values())) if tests else set()
    intersections = {
        "train_validation": sorted(train_set & validation_set),
        "train_test": sorted(train_set & test_union),
        "validation_test": sorted(validation_set & test_union),
    }
    if any(intersections.values()):
        raise ValueError(
            "Gene-level split leakage detected: "
            + ", ".join(f"{key}={len(value)}" for key, value in intersections.items())
        )
    return {
        "schema_version": 1,
        "population": population,
        "seed": int(seed),
        "split_unit": "gene",
        "gene_universe_ordering_before_split": "sorted Ensembl gene ID",
        "train_genes": train_genes,
        "validation_genes": validation_genes,
        "test_genes_by_task": tests,
        "counts": {
            "train": len(train_genes),
            "validation": len(validation_genes),
            "test_union": len(test_union),
            "test_by_task": {task: len(genes) for task, genes in tests.items()},
        },
        "set_sha256": {
            "train": _gene_list_hash(train_genes),
            "validation": _gene_list_hash(validation_genes),
            "test_union": _gene_list_hash(test_union),
            "test_by_task": {task: _gene_list_hash(genes) for task, genes in tests.items()},
        },
        "intersections": {key: len(value) for key, value in intersections.items()},
    }


def train_model(data, output_dir=None, frozen_split_manifest=None):
    """Train a Varformer model for one (population, seed) configuration."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    config = data['config']
    set_seed(config['hyperparameters']['seed'])
    started_at = _clock_now()
    run_name = config['hyperparameters'].get('run_name')

    run = wandb.init(
        project="drug-target-prediction",
        config=config['hyperparameters'],
        group=f"varformer-{config['hyperparameters']['population']}",
        name=run_name,
        mode=None if config['hyperparameters'].get('wandb', True) else "disabled",
    )

    preprocessor = ModelPreprocessorEval(
        config, data, frozen_split_manifest=frozen_split_manifest
    )
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params})

    # Setup training callbacks
    if output_dir is None:
        current_date = _clock_date_directory()
        checkpoint_dir = Path("checkpoints") / current_date
    else:
        checkpoint_dir = Path(output_dir) / config['hyperparameters']['population']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = _split_manifest_payload(
        preprocessor,
        config['hyperparameters']['population'],
        int(config['hyperparameters']['seed']),
    )
    if preprocessor.frozen_split_source is not None:
        if split_manifest["set_sha256"] != preprocessor.frozen_split_source["set_sha256"]:
            raise ValueError("Materialized split differs from the validated frozen source")
        split_manifest["frozen_split_source"] = preprocessor.frozen_split_source
    split_manifest_path = checkpoint_dir / f"seed{config['hyperparameters']['seed']}.split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_spearman',
        dirpath=str(checkpoint_dir),
        filename=f"seed{config['hyperparameters']['seed']}" + '-{epoch:02d}-{val_spearman:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    best_threshold_callback = BestThresholdCallback(
        monitor='val_spearman',
        mode='max'
    )

    nan_diagnostics_callback = NaNDiagnosticsCallback(
        stop_on_nan=bool(config['hyperparameters'].get('stop_on_nan', True)),
    )

    callbacks = [lr_monitor, checkpoint_callback, best_threshold_callback, nan_diagnostics_callback]

    set_seed(config['hyperparameters']['seed'])

    # Build trainer kwargs, adding conditional options only when needed.
    trainer_kwargs = dict(
        max_epochs=int(config['hyperparameters']['epochs']),
        accelerator=accelerator,
        enable_progress_bar=True,
        precision=config['hyperparameters']['precision'],
        logger=WandbLogger(wandb.run),
        callbacks=callbacks,
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

    validation_predictions_path = None
    validation_prediction_count = None
    if bool(config['hyperparameters'].get('use_pvc', True)):
        # Export predictions from the best checkpoint on validation genes only.
        # These paired records provide the uncertainty calculation for token
        # schema selection without opening any held-out benchmark.
        validation_batches = trainer.predict(dataloaders=val_combined, ckpt_path='best')
        validation_records = []
        for batch in validation_batches:
            for gene_id, values in batch.items():
                validation_records.append({
                    "gene_id": gene_id,
                    "prediction": float(values["prediction"]),
                    "label": int(values["label"]),
                })
        validation_predictions_path = (
            checkpoint_dir / f"seed{config['hyperparameters']['seed']}.validation_predictions.csv"
        )
        with validation_predictions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["gene_id", "prediction", "label"])
            writer.writeheader()
            writer.writerows(validation_records)
        validation_prediction_count = len(validation_records)

    holdout_results = {}
    evaluate_holdout = bool(config['hyperparameters'].get('evaluate_holdout_after_fit', True))
    if evaluate_holdout:
        print("Starting held-out benchmark evaluation...")
        for benchmark in ("pfam", "rcnt", "pharos"):
            holdout_results[benchmark] = trainer.test(
                dataloaders=test_combined[benchmark], ckpt_path='best'
            )
    else:
        print("Validation-only run: held-out benchmarks were not evaluated.")

    wandb_run_id = run.id
    run.finish()

    best_path = checkpoint_callback.best_model_path
    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "population": config['hyperparameters']['population'],
        "seed": int(config['hyperparameters']['seed']),
        "configuration": _jsonable(config['hyperparameters']),
        "configuration_sha256": _configuration_hash(config['hyperparameters']),
        "selection_metric": "val_spearman",
        "best_validation_score": _jsonable(checkpoint_callback.best_model_score),
        "best_validation_threshold": _jsonable(best_threshold_callback.best_threshold),
        "split_manifest_path": str(split_manifest_path),
        "split_set_sha256": split_manifest["set_sha256"],
        "split_counts": split_manifest["counts"],
        "frozen_split_source": preprocessor.frozen_split_source,
        "nnpu_correction_history": _jsonable(model.nnpu_correction_history),
        "nnpu_correction_active_batches": int(sum(
            item["active_batches"] for item in model.nnpu_correction_history
        )),
        "nnpu_correction_total_batches": int(sum(
            item["total_batches"] for item in model.nnpu_correction_history
        )),
        "nnpu_correction_ever_active": bool(any(
            item["active_batches"] > 0 for item in model.nnpu_correction_history
        )),
        "validation_predictions_path": str(validation_predictions_path)
        if validation_predictions_path else None,
        "validation_prediction_count": validation_prediction_count,
        "best_checkpoint_path": best_path or None,
        "last_checkpoint_path": checkpoint_callback.last_model_path or None,
        "heldout_evaluated": evaluate_holdout,
        "holdout_results": _jsonable(holdout_results),
        "wandb_run_id": wandb_run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": started_at,
        "completed_at": _clock_now(),
        "timestamp_source": "runtime-host date command: date '+%Y-%m-%dT%H:%M:%S%z'",
        "status": "COMPLETED",
    }
    manifest_path = checkpoint_dir / f"seed{config['hyperparameters']['seed']}.run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run manifest: {manifest_path}")
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

    def __init__(
        self,
        population: str,
        config_overrides=None,
        output_dir=None,
        frozen_split_manifest=None,
    ):
        from varformer.config import Config
        self.population = population
        self.config = Config.load(hyperparams_override=config_overrides or {})
        self.output_dir = output_dir
        self.frozen_split_manifest = frozen_split_manifest

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
            # Seed before preprocessing and model construction so the requested
            # seed controls splits, initialization, and loader shuffling.
            set_seed(int(seed))
            cfg = {
                "hyperparameters": {
                    **self.config.hyperparameters.model_dump(),
                    "seed": int(seed),
                    "population": self.population,
                    "mode": "eval",
                },
                "paths": self.config.paths,
            }
            # Load all modalities even for ablations. This freezes the eligible
            # gene universe and split identities across the model matrix.
            data = ModuleDataProcessor(gc=True, go=True, pvc=True, config=cfg).process()
            training_kwargs = {"output_dir": self.output_dir}
            if self.frozen_split_manifest is not None:
                training_kwargs["frozen_split_manifest"] = self.frozen_split_manifest
            ckpt_path = train_model(data, **training_kwargs)
            if ckpt_path is not None:
                paths.append(Path(ckpt_path))
        return paths
