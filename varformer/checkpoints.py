"""Checkpoint utilities: loading published checkpoints and discovering them on disk."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


_METRIC_KEY_RE = re.compile(r"^model\.(acc|auroc|recall|precision|f1|auprc|spearman)\.")
_DEAD_CLASSIFIER_RE = re.compile(r"^model\.varformer\.classifier\.")
_WRAPPER_PREFIX = "model.varformer.varformer."
_WRAPPER_NEW_PREFIX = "model.varformer."


def _remap_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remap a saved state_dict onto the current model's parameter names.

    1. Drop metric keys (``model.acc.*``, ``model.auroc.*``, ...): metrics live
       on the LightningModule, not the model.
    2. Drop ``model.varformer.classifier.*`` keys, which have no counterpart in
       the current model.
    3. Rename ``model.varformer.varformer.X`` -> ``model.varformer.X``.
    """
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if _METRIC_KEY_RE.match(k):
            continue
        if _DEAD_CLASSIFIER_RE.match(k):
            continue
        if k.startswith(_WRAPPER_PREFIX):
            k = _WRAPPER_NEW_PREFIX + k[len(_WRAPPER_PREFIX):]
        out[k] = v
    return out


def load_checkpoint(ckpt_path: str | Path) -> dict:
    """Load a Lightning checkpoint, remapping state_dict keys for compatibility.

    Returns the full Lightning checkpoint dict.  Callers should do:
    ``lm.load_state_dict(ckpt['state_dict'], strict=False)``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt["state_dict"] = _remap_state_dict(ckpt["state_dict"])
    return ckpt


def find_checkpoint(ckpt_root: Path, population: str, seed: int) -> Path:
    pattern = f"seed{seed}-epoch=*-val_spearman=*.ckpt"
    matches = list((ckpt_root / population).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint for (pop={population}, seed={seed}) in {ckpt_root}, got {len(matches)}"
        )
    return matches[0]


def list_checkpoints(ckpt_root: Path, population: str) -> list[Path]:
    return sorted((ckpt_root / population).glob("seed*-epoch=*-val_spearman=*.ckpt"))


def best_seed(ckpt_root: Path, population: str) -> int:
    best = (None, -1.0)
    for p in list_checkpoints(ckpt_root, population):
        m = re.search(r"seed(\d+)-epoch=\d+-val_spearman=([\d.]+)", p.name)
        if m:
            seed_, sp = int(m.group(1)), float(m.group(2))
            if sp > best[1]:
                best = (seed_, sp)
    if best[0] is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_root / population}")
    return best[0]
