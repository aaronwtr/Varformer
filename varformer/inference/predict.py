"""Run subset predictions for genes from a pre-loaded Varformer instance.

``predict_subset`` drives the SDK's ``Varformer.predict()`` method.  It reuses
the Lightning module and test-data loaders cached at model-load time, so outputs
are bit-exact with the benchmark reference predictions.
"""
import torch

from pytorch_lightning import Trainer

from varformer.training.lightning_module import VarformerLightningModule


def predict_subset(model, genes, return_attention=False):
    """Run inference for the SDK's predict() method.

    Uses the LightningModule and config cached on ``model`` by
    ``Varformer._build_and_load`` and mirrors the data pipeline used in
    ``benchmark/generate_reference.py:generate_for_population`` so that outputs
    are bit-exact with the benchmark reference.

    Args:
        model:            A ``Varformer`` nn.Module instance with ``_lightning_module``,
                          ``_config``, and ``_population`` attributes set by
                          ``Varformer._build_and_load``.
        genes:            List of Ensembl gene IDs to return predictions for.
        return_attention: If False, strip ``attn_weights`` from returned payloads.

    Returns:
        dict mapping gene_id -> {"prediction", "classification", "z_var"[, "attn_weights"]}
    """
    lm = model._lightning_module
    # Reuse test_loaders cached at model-load time; they were built from the same data
    # pipeline + config that produced the benchmark reference predictions.
    test_loaders = model._test_loaders

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    gene_set = set(genes)
    results: dict = {}
    for loader_name, loader in test_loaders.items():
        batch_results = trainer.predict(model=lm, dataloaders=loader)
        for batch in batch_results:
            for gid, payload in batch.items():
                if gid in gene_set:
                    results[gid] = payload

    # Strip attn_weights if not requested.
    if not return_attention:
        for payload in results.values():
            payload.pop("attn_weights", None)

    return results
