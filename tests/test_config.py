"""Tests for varformer.config — Pydantic Config + Hyperparameters + Paths."""
import yaml

from varformer.config import CONFIGS, PACKAGE_CONFIGS, Config, Hyperparameters


def test_packaged_defaults_match_source_defaults():
    """Prevent the installed-wheel fallback from drifting from source config."""
    with (CONFIGS / "default.yml").open() as source_file:
        source = yaml.safe_load(source_file)
    with (PACKAGE_CONFIGS / "default.yml").open() as packaged_file:
        packaged = yaml.safe_load(packaged_file)
    assert packaged == source


def test_load_default_local():
    c = Config.load(profile="local")
    assert c.hyperparameters.d_model == 256
    assert c.hyperparameters.max_seq_len == 1024
    assert c.hyperparameters.optimizer == "AdamW"


def test_load_hpc_profile():
    # The hpc profile resolves (to hpc.yml if present, else hpc.example.yml)
    # and yields both path roots.
    c = Config.load(profile="hpc")
    assert c.paths.data_root is not None
    assert c.paths.ckpt_root is not None


def test_missing_profile_falls_back_to_example():
    # A profile with no real <profile>.yml uses the tracked <profile>.example.yml.
    c = Config.load(profile="local")
    assert c.paths.data_root is not None


def test_hyperparams_override():
    c = Config.load(
        profile="local",
        hyperparams_override={"epochs": 5, "batch_size": 64, "optimizer_foreach": False},
    )
    assert c.hyperparameters.epochs == 5
    assert c.hyperparameters.batch_size == 64
    assert c.hyperparameters.optimizer_foreach is False
    assert c.hyperparameters.d_model == 256  # untouched


def test_dict_access_hyperparameters():
    c = Config.load(profile="local")
    assert c["hyperparameters"]["d_model"] == 256
    assert c["hyperparameters"]["max_seq_len"] == 1024


def test_dict_access_paths():
    c = Config.load(profile="local")
    assert "checkpoints" in c["paths"]["CKPT_PATH"]
    assert "features" in c["paths"]["FEATURES_DIR"]
    assert "missense_mutation_map.pkl" in c["paths"]["MISSENSE_MAP"]
    assert c["paths"]["FDA_LABELS"].endswith("FDA_approved_drug_targets_2023_Q3.xlsx")


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
