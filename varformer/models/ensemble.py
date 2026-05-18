"""Ensemble wrapper that averages predictions across all 5 seeds of a population."""
from __future__ import annotations

import numpy as np


class VarformerEnsemble:
    """Ensemble wrapper that averages predictions across all seeds of a population.

    Wraps N ``Varformer`` instances (one per training seed) and combines their
    outputs by arithmetic mean.  Returned by
    ``Varformer.from_pretrained(population, seed="ensemble")``.

    Note:
        ``evaluate()`` delegates to the **first** model instance only; it does
        not run evaluation on all seeds and therefore does not reflect true
        ensemble performance on the holdout set.

    Example:
        >>> ensemble = Varformer.from_pretrained("nfe", seed="ensemble")
        >>> predictions = ensemble.predict(["ENSG00000141510", "ENSG00000012048"])
        >>> print(predictions["ENSG00000141510"]["prediction"])
        0.76
    """

    def __init__(self, models: list):
        if not models:
            raise ValueError("Empty ensemble")
        self._models = models

    @classmethod
    def from_pretrained(cls, population: str):
        """Load all available checkpoints for a population as an ensemble.

        Discovers every checkpoint under the population's checkpoint directory
        via ``varformer.checkpoints.list_checkpoints`` and builds one
        ``Varformer`` instance per checkpoint using
        ``Varformer.from_checkpoint``.

        Args:
            population: Population identifier.  One of ``"nfe"``, ``"elgh"``,
                ``"afr"``, ``"amr"``.

        Returns:
            A ``VarformerEnsemble`` containing one model per discovered
            checkpoint.

        Raises:
            ValueError: if no checkpoints are found for the given population
                (the resulting empty ensemble is rejected by ``__init__``).

        Example:
            >>> ensemble = VarformerEnsemble.from_pretrained("nfe")
            >>> print(len(ensemble._models))
            5
        """
        from varformer.checkpoints import list_checkpoints
        from varformer.config import Config
        from varformer.models.varformer import Varformer

        config = Config.load()
        paths = list_checkpoints(config.paths.ckpt_root, population)
        models = [Varformer.from_checkpoint(p) for p in paths]
        return cls(models)

    def predict(self, genes, return_attention: bool = False) -> dict:
        """Run ensemble inference and return averaged predictions.

        Calls ``predict()`` on every member model, then averages probabilities
        and ``z_var`` embeddings element-wise.  Classification is determined
        by whether the mean probability exceeds ``0.5`` (not majority vote).

        Args:
            genes: List of Ensembl gene IDs to return predictions for.  Genes
                not present in any member model's loaders are silently omitted.
            return_attention: When ``True``, include per-variant attention
                weights averaged across all member models.

        Returns:
            A dict mapping each recognised gene ID to a payload dict with the
            following keys:

            * ``"prediction"`` (``float`` in ``[0, 1]``) — mean sigmoid
              probability across all seeds.
            * ``"classification"`` (``int``, ``0`` or ``1``) — ``1`` when
              ``mean_p > 0.5``, else ``0``.
            * ``"z_var"`` (``numpy.ndarray`` of shape ``(d_model,)``) — element-
              wise mean of the attended variant embeddings across seeds.
            * ``"attn_weights"`` (``numpy.ndarray`` of shape ``(max_seq_len,)``)
              — element-wise mean of per-variant attention weights across seeds.
              **Only present when** ``return_attention=True``.

        Example:
            >>> ensemble = Varformer.from_pretrained("nfe", seed="ensemble")
            >>> preds = ensemble.predict(["ENSG00000141510"])
            >>> print(preds["ENSG00000141510"]["classification"])
            1
        """
        per_seed = [m.predict(genes, return_attention=return_attention) for m in self._models]
        if not per_seed:
            return {}

        out: dict = {}
        for gid in per_seed[0]:
            probs = np.array([s[gid]["prediction"] for s in per_seed])
            mean_p = float(probs.mean())
            entry = {
                "prediction": mean_p,
                "classification": int(mean_p > 0.5),
                "z_var": np.mean(np.stack([s[gid]["z_var"] for s in per_seed]), axis=0),
            }
            if return_attention:
                entry["attn_weights"] = np.mean(
                    np.stack([s[gid]["attn_weights"] for s in per_seed]), axis=0
                )
            out[gid] = entry
        return out

    def evaluate(self, test_set: str) -> dict:
        """Evaluate the ensemble on a labelled holdout set.

        Delegates entirely to the **first** member model.  This is not a true
        ensemble evaluation — only seed 0's predictions are scored.  Use
        ``predict()`` followed by a custom metric computation if full ensemble
        evaluation is required.

        Args:
            test_set: Which labelled test partition to score against.  One of:
                ``"pfam"``   — Pfam-derived holdout genes.
                ``"rcnt"``   — Recent FDA-approval holdout genes.
                ``"pharos"`` — Pharos chemoinformatics holdout genes.

        Returns:
            Metrics dict from the first member model's ``evaluate()`` call,
            with keys such as ``"auroc"``, ``"auprc"``, ``"spearman"``,
            ``"accuracy"``, ``"recall"``, ``"precision"``, ``"f1"``.

        Raises:
            KeyError: if ``test_set`` is not one of the three supported names.
        """
        # Delegate to first instance for simplicity.
        return self._models[0].evaluate(test_set)
