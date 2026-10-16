"""Tests for loading the released weights (local copy of the release layout)."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from varformer.checkpoints import RELEASED_MODELS, load_released


def _release(tmp_path, folder="nfe", seeds=(13, 42), spearman=(0.40, 0.41)):
    index = []
    for seed, score in zip(seeds, spearman):
        path = tmp_path / folder / f"seed{seed}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"model.varformer.w": torch.full((2,), float(seed)), "model.acc.tp": torch.zeros(1)}, str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / folder / f"seed{seed}.json").write_text(json.dumps({"weights_sha256": digest, "seed": seed}))
        index.append({"file": f"{folder}/seed{seed}.safetensors", "seed": seed, "validation_spearman": score})
    (tmp_path / "index.json").write_text(json.dumps(index))
    return tmp_path


def test_load_released_seed_and_remap(tmp_path):
    population, state, description = load_released("nfe", seed=13, local_dir=_release(tmp_path))
    assert population == "nfe" and description["seed"] == 13
    assert torch.equal(state["model.varformer.w"], torch.full((2,), 13.0))
    assert "model.acc.tp" not in state


def test_best_seed_uses_validation_spearman(tmp_path):
    _, _, description = load_released("nfe", seed="best", local_dir=_release(tmp_path))
    assert description["seed"] == 42


def test_sas_gnh_uses_sas_inputs(tmp_path):
    assert RELEASED_MODELS["sas-gnh"] == "sas"
    population, _, _ = load_released("sas-gnh", seed=13, local_dir=_release(tmp_path, folder="sas-gnh"))
    assert population == "sas"


def test_checksum_mismatch_is_rejected(tmp_path):
    root = _release(tmp_path)
    (root / "nfe" / "seed13.json").write_text(json.dumps({"weights_sha256": "0" * 64, "seed": 13}))
    with pytest.raises(ValueError, match="Checksum"):
        load_released("nfe", seed=13, local_dir=root)


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        load_released("eas", local_dir=tmp_path)
