"""Tests for varformer.config — Pydantic Config + Hyperparameters + Paths."""
import pytest
from varformer.config import Config, Hyperparameters, Paths


def test_load_default_local():
    c = Config.load(profile="local")
    assert c.hyperparameters.d_model == 256
    assert c.hyperparameters.max_seq_len == 1024
    assert c.hyperparameters.optimizer == "AdamW"


def test_load_hpc():
    c = Config.load(profile="hpc")
    assert "/gpfs/scratch" in str(c.paths.data_root)
    assert "/data/home" in str(c.paths.ckpt_root)


def test_hyperparams_override():
    c = Config.load(profile="local", hyperparams_override={"epochs": 5, "batch_size": 64})
    assert c.hyperparameters.epochs == 5
    assert c.hyperparameters.batch_size == 64
    assert c.hyperparameters.d_model == 256  # untouched


def test_legacy_dict_access_hyperparameters():
    c = Config.load(profile="local")
    # Legacy code does config['hyperparameters']['d_model']
    assert c["hyperparameters"]["d_model"] == 256
    assert c["hyperparameters"]["max_seq_len"] == 1024


def test_legacy_dict_access_paths():
    c = Config.load(profile="local")
    # Legacy code does config['paths']['CKPT_PATH']
    assert "checkpoints" in c["paths"]["CKPT_PATH"]
    assert "features" in c["paths"]["FEATURES_DIR"]
    assert "missense_mutation_map.pkl" in c["paths"]["MISSENSE_MAP"]


def test_hyperparams_contains_and_get():
    hp = Hyperparameters()
    assert "d_model" in hp
    assert "nonexistent" not in hp
    assert hp.get("d_model") == 256
    assert hp.get("nonexistent", "default") == "default"


def test_hyperparams_setitem():
    hp = Hyperparameters()
    hp["epochs"] = 99
    assert hp.epochs == 99
