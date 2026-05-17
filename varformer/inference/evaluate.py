"""Model evaluation and testing utilities for the Varformer model.

Moved from src/testing.py in Phase 5.
Phase 7: Added evaluate_subset() for SDK evaluate() method.
"""
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from varformer.training.lightning_module import VarformerLightningModule
from varformer.data.pipeline import ModuleDataProcessor
from varformer.data.loaders import ModelPreprocessorEval


def load_test_data(go, gc, pvc, psc, config):
    data = ModuleDataProcessor(gc, go, pvc, psc, config).process()
    return preprocess_test_data(data, config)


def preprocess_test_data(data, config):
    preprocessor = ModelPreprocessorEval(config, data)
    _, _, _, test_combined, _, _ = preprocessor.model_init()
    return test_combined


def load_model(config, data):
    preprocessor = ModelPreprocessorEval(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()
    return model


def test_model(test_data, model_checkpoint_path, config):
    model = VarformerLightningModule.load_from_checkpoint(model_checkpoint_path, config=config)
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.test(model=model, dataloaders=test_data["pfam"])
    trainer.test(model=model, dataloaders=test_data["rcnt"])
    trainer.test(model=model, dataloaders=test_data["pharos"])


def extract_and_save_test_genes(test_data, population):
    """
    Extracts gene names from the test data and saves them to a file.
    """
    # Extract unique gene names from the test data
    gene_names = set()

    for loader_name, loader in test_data.items():
        print(f"Processing {loader_name}...")

        for dataset_name, dataset in loader.datasets.items():
            if hasattr(dataset, 'gene_names') and dataset.gene_names is not None:
                print(f"  {dataset_name}: Found {len(dataset.gene_names)} genes")
                gene_names.update(dataset.gene_names)
            else:
                print(f"  {dataset_name}: No gene_names or gene_names is None")

    if gene_names:
        file_path = f"test_genes_{population}.txt"
        with open(file_path, "w") as f:
            for gene in sorted(list(gene_names)):
                f.write(f"{gene}\n")
        print(f"Test gene names saved to {file_path}")


def run_test(**modules):
    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    config = modules.get('config', {})
    extract_genes_only = modules.get('extract_genes_only', False)

    test_data = load_test_data(gc, go, pvc, psc, config)

    population = config.get('hyperparameters', {}).get('population', 'default')
    extract_and_save_test_genes(test_data, population)

    if extract_genes_only:
        print("Gene extraction complete. Skipping model testing.")
        return

    model_ckpt_path = 'checkpoints/seed42-epoch=98-val_auroc=0.87.ckpt'
    test_model(test_data, model_ckpt_path, config)


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
