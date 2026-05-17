"""Ensemble wrapper that averages predictions across all 5 seeds of a population."""
from __future__ import annotations

import numpy as np


class VarformerEnsemble:
    def __init__(self, models: list):
        if not models:
            raise ValueError("Empty ensemble")
        self._models = models

    @classmethod
    def from_pretrained(cls, population: str):
        from varformer.checkpoints import list_checkpoints
        from varformer.config import Config
        from varformer.models.varformer import Varformer

        config = Config.load()
        paths = list_checkpoints(config.paths.ckpt_root, population)
        models = [Varformer.from_checkpoint(p) for p in paths]
        return cls(models)

    def predict(self, genes, return_attention: bool = False) -> dict:
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
        # Delegate to first instance for simplicity.
        return self._models[0].evaluate(test_set)
