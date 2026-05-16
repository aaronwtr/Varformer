"""Tests for VarformerLightningModule."""
import pytest
import torch
from varformer.training.lightning_module import VarformerLightningModule


@pytest.fixture
def tiny_config(tiny_hyperparams):
    return {"hyperparameters": tiny_hyperparams}


@pytest.fixture
def tiny_lm(tiny_config):
    return VarformerLightningModule(
        config=tiny_config,
        num_samples_per_class=None,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        class_prior=0.1,
        use_pvc=True,
    )


def test_instantiation(tiny_lm):
    assert tiny_lm.model is not None
    assert hasattr(tiny_lm, "acc")
    assert hasattr(tiny_lm, "auroc")
    assert hasattr(tiny_lm, "spearman")
    assert hasattr(tiny_lm, "recall")
    assert hasattr(tiny_lm, "precision")
    assert hasattr(tiny_lm, "f1")
    assert hasattr(tiny_lm, "auprc")


def test_metrics_not_on_model(tiny_lm):
    """Metrics must live on the LightningModule, NOT inside model."""
    model_children = {name for name, _ in tiny_lm.model.named_children()}
    assert "acc" not in model_children
    assert "auroc" not in model_children
    assert "spearman" not in model_children


def test_forward_shape(tiny_lm):
    B, S = 4, 10
    x = {
        "gc": (torch.rand(B, 30),),
        "go": (torch.rand(B, 20),),
        "pvc": {
            "pathogenicity": torch.rand(B, S),
            "position": torch.randint(0, 16, (B, S)),
            "mutation": torch.randint(0, 50, (B, S)),
        },
    }
    mask = torch.zeros(B, S, dtype=torch.bool)
    out = tiny_lm(x, mask)
    assert len(out) == 5  # return_attn=True by default in tiny_hyperparams


def test_model_attribute_names(tiny_lm):
    """Key model attribute names must be preserved for checkpoint loading."""
    names = {n for n, _ in tiny_lm.model.named_children()}
    assert "gc_projection" in names
    assert "go_projection" in names
    assert "varformer" in names
    assert "gene_variant_attention" in names
    assert "classification_head" in names


def test_state_dict_has_no_metric_keys(tiny_lm):
    """State_dict keys under 'model.' must not include metric names."""
    sd = tiny_lm.model.state_dict()
    metric_prefixes = ("acc.", "auroc.", "recall.", "precision.", "f1.", "auprc.", "spearman.")
    for k in sd:
        assert not any(k.startswith(p) for p in metric_prefixes), f"Unexpected metric key in model: {k}"


def test_metrics_registered_as_submodules(tiny_lm):
    """Metrics must be registered as named sub-modules on the LightningModule."""
    children = {name for name, _ in tiny_lm.named_children()}
    assert "acc" in children, "acc not a child of LM"
    assert "spearman" in children, "spearman not a child of LM"
    assert "auroc" in children, "auroc not a child of LM"
