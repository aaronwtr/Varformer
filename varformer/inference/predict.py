"""Run subset predictions for genes from a pre-loaded Varformer instance.

``predict_subset`` drives the SDK's ``Varformer.predict()`` method.  It reuses
the Lightning module and test-data loaders cached at model-load time.
"""
import torch

from pytorch_lightning import Trainer



def predict_subset(model, genes, return_attention=False):
    """Run inference for the SDK's predict() method.

    Uses the LightningModule and data loaders cached on ``model`` by
    ``Varformer._build_and_load``.

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


def score_variants_subset(model, genes=None):
    """Score individual missense variants with the variant head.

    Uses the loaders cached on ``model`` by ``Varformer._build_and_load``.

    Args:
        model: A ``Varformer`` instance trained with ``variant_head=True``.
        genes: Ensembl gene IDs to score, or ``None`` for every cached gene.

    Returns:
        dict mapping gene_id -> {"variant_scores", "pathogenicity", "position",
        "mutation"}. Each value is a NumPy array with one entry per retained
        variant, in the model's token order.
    """
    if model.variant_head is None:
        raise ValueError("This model was trained without a variant head")

    gene_set = None if genes is None else set(genes)
    results: dict = {}
    model.eval()
    with torch.no_grad():
        for loader in model._test_loaders.values():
            for batch in loader:
                pvc = batch["pvc"]
                output = model({"gc": batch["gc"], "go": batch["go"], "pvc": pvc}, pvc["mask"])
                scores = output[-1].float().cpu().numpy()
                real = ~pvc["mask"].bool().cpu().numpy()
                for i, gid in enumerate(pvc["gene_name"]):
                    if gene_set is not None and gid not in gene_set:
                        continue
                    results[gid] = {
                        "variant_scores": scores[i][real[i]],
                        **{
                            key: pvc[key][i].cpu().numpy()[real[i]]
                            for key in ("pathogenicity", "position", "mutation")
                        },
                    }
    return results
