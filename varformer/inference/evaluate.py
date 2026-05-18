"""Evaluate a pre-loaded Varformer instance on labelled holdout sets.

``evaluate_subset`` drives the SDK's ``Varformer.evaluate()`` method.  It runs
``trainer.test()`` on one of the three named holdout partitions (pfam, rcnt,
pharos) and returns a normalised metric dict.
"""
import torch

from pytorch_lightning import Trainer

from varformer.training.lightning_module import VarformerLightningModule


def evaluate_subset(model, test_set: str) -> dict:
    """Run evaluation on a labelled holdout set and return metric dict.

    Uses the LightningModule and config cached on ``model`` by
    ``Varformer._build_and_load``.  Runs ``trainer.test()`` on the requested
    test_set and returns normalised metric names (``test_<source>_`` prefix
    stripped).

    As a fallback, if the test pipeline raises, metrics are computed manually
    from ``predict_subset`` output using sklearn.

    Args:
        model:    A ``Varformer`` nn.Module instance with ``_lightning_module``,
                  ``_config``, and ``_population`` attributes.
        test_set: One of "pfam", "rcnt", "pharos".

    Returns:
        dict with keys like "auroc", "auprc", "spearman", "acc", etc.
    """
    import pandas as pd
    import sys
    from pathlib import Path

    lm = model._lightning_module
    config = model._config
    population = model._population

    cfg = {
        "hyperparameters": {
            **config.hyperparameters.model_dump(),
            "population": population,
            "return_attn": True,
            "mode": "inference",
        },
        "paths": config.paths.legacy,
    }

    from varformer.data.pipeline import ModuleDataProcessor
    from varformer.data.loaders import ModelPreprocessorInference

    data = ModuleDataProcessor(gc=True, go=True, pvc=True, psc=False, config=cfg).process()
    splits = data if isinstance(data, list) else [data]
    first = splits[0]

    consolidated_data = {
        "gc": pd.concat([first["train"]["gc"], first["test_data"]["gc"]]),
        "go": pd.concat([first["train"]["go"], first["test_data"]["go"]]),
    }
    consolidated_pvc = {**first["train"]["pvc"], **first["test_data"]["pvc"]}
    consolidated_pvc.pop("labels", None)

    test_loaders = ModelPreprocessorInference.create_test_loaders(
        config=cfg,
        consolidated_data=consolidated_data,
        pvc_data=consolidated_pvc,
        torch_dtype=cfg["hyperparameters"]["precision"],
    )

    if test_set not in test_loaders:
        raise ValueError(
            f"test_set '{test_set}' not found. Available: {list(test_loaders.keys())}"
        )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    metrics_list = trainer.test(model=lm, dataloaders=test_loaders[test_set])
    if not metrics_list:
        return {}

    raw = metrics_list[0]
    # Normalize keys: strip "test_<source>_" prefix, e.g. "test_pfam_auroc" -> "auroc"
    prefix = f"test_{test_set}_"
    out = {}
    for k, v in raw.items():
        key = k[len(prefix):] if k.startswith(prefix) else k
        out[key] = float(v) if hasattr(v, "__float__") else v
    return out
