"""Load the released Varformer weights and locally trained, trusted checkpoints."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


HUB_REPO_ID = "aaronw1290/Varformer"
# Released model -> population of its input data. sas-gnh: the SAS models with
# geneticConstraint as an input, used for the Genes & Health variant analyses.
RELEASED_MODELS = {"afr": "afr", "amr": "amr", "nfe": "nfe", "sas": "sas", "sas-gnh": "sas"}

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

    Lightning checkpoints are executable serialization formats. Only load a
    checkpoint obtained from a trusted, checksum-verified source.

    Returns the full Lightning checkpoint dict.  Callers should do:
    ``lm.load_state_dict(ckpt['state_dict'], strict=False)``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt["state_dict"] = _remap_state_dict(ckpt["state_dict"])
    return ckpt


def saved_gc_exclusions(ckpt: dict, default: list[str]) -> list[str]:
    """Return the gene features that a checkpoint was trained without."""
    saved = ckpt.get("hyper_parameters", {}).get("config")
    if not isinstance(saved, dict) or "hyperparameters" not in saved:
        return list(default)
    # Checkpoints saved before the option existed used every feature.
    return list(saved["hyperparameters"].get("gc_exclude_features", []))


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


def _release_file(name: str, repo_id: str, revision: str | None, local_dir: str | Path | None) -> Path:
    if local_dir is not None:
        path = Path(local_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found in {local_dir}")
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=name, revision=revision))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_released(
    model: str,
    seed: int | str = 42,
    repo_id: str = HUB_REPO_ID,
    revision: str | None = None,
    local_dir: str | Path | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return ``(population, state_dict, description)`` for a released model.

    Files come from the Hugging Face Hub, or from ``local_dir`` when it holds a
    copy of the release. ``seed="best"`` selects the seed with the highest
    validation Spearman correlation. The weights are checked against the
    SHA-256 checksum in their description.
    """
    if model not in RELEASED_MODELS:
        raise ValueError(f"Unknown model {model!r}; choose one of {sorted(RELEASED_MODELS)}")
    if seed == "best":
        index = json.loads(_release_file("index.json", repo_id, revision, local_dir).read_text())
        entries = [e for e in index if e["file"].startswith(f"{model}/")]
        if not entries:
            raise FileNotFoundError(f"No released seeds for {model!r}")
        seed = max(entries, key=lambda e: e["validation_spearman"])["seed"]
    stem = f"{model}/seed{int(seed)}"
    description = json.loads(_release_file(f"{stem}.json", repo_id, revision, local_dir).read_text())
    weights = _release_file(f"{stem}.safetensors", repo_id, revision, local_dir)
    if _sha256(weights) != description["weights_sha256"]:
        raise ValueError(f"Checksum mismatch for {stem}.safetensors")
    from safetensors.torch import load_file

    return RELEASED_MODELS[model], _remap_state_dict(load_file(str(weights))), description
