"""Tests for Varformer (renamed from MultiModalTargetIdentifier)."""
import pytest
import torch
from varformer.models.varformer import Varformer


@pytest.fixture
def tiny_config(tiny_hyperparams):
    """Minimal config dict that mimics legacy dict access."""
    return {"hyperparameters": tiny_hyperparams}


@pytest.fixture
def tiny_model(tiny_config):
    return Varformer(
        config=tiny_config,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        use_pvc=True,
    )


def _make_batch(B=4, S=10, num_gc=30, num_go=20, num_muts=50):
    gc_feat = torch.rand(B, num_gc)
    go_feat = torch.rand(B, num_go)
    path = torch.rand(B, S)
    pos = torch.randint(0, 16, (B, S))
    mut = torch.randint(0, num_muts, (B, S))
    mask = torch.zeros(B, S, dtype=torch.bool)
    return {
        "gc": (gc_feat,),
        "go": (go_feat,),
        "pvc": {
            "pathogenicity": path,
            "position": pos,
            "mutation": mut,
        },
    }, mask


def test_forward_returns_5_with_return_attn(tiny_model):
    x, mask = _make_batch()
    out = tiny_model(x, mask)
    assert len(out) == 5, f"Expected 5-tuple, got {len(out)}"
    logits, probas, bin_preds, z_var, attn_w = out
    B = 4
    assert logits.shape == (B,)
    assert probas.shape == (B,)
    assert bin_preds.shape == (B,)
    assert z_var is not None
    assert attn_w is not None


def test_forward_returns_4_without_return_attn(tiny_config):
    tiny_config["hyperparameters"]["return_attn"] = False
    model = Varformer(
        config=tiny_config,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        use_pvc=True,
    )
    x, mask = _make_batch()
    out = model(x, mask)
    assert len(out) == 4, f"Expected 4-tuple, got {len(out)}"


def test_no_pvc_branch(tiny_config):
    tiny_config["hyperparameters"]["return_attn"] = False
    model = Varformer(
        config=tiny_config,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        use_pvc=False,
    )
    B = 3
    x = {
        "gc": (torch.rand(B, 30),),
        "go": (torch.rand(B, 20),),
    }
    out = model(x, mask=None)
    assert len(out) == 4
    logits, probas, bin_preds, z_var = out
    assert z_var is None


def test_attribute_names_for_checkpoint(tiny_model):
    """Key attribute names must match checkpoint state_dict keys."""
    names = {name for name, _ in tiny_model.named_children()}
    assert "gc_projection" in names
    assert "go_projection" in names
    assert "varformer" in names            # VariantEncoder lives here
    assert "gene_variant_attention" in names
    assert "classification_head" in names


def test_config_dict_access(tiny_config):
    """Config can be a plain dict (unit test path)."""
    model = Varformer(
        config=tiny_config,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
    )
    assert model.hyperparams["d_model"] == 32


def test_probas_in_unit_interval(tiny_model):
    x, mask = _make_batch()
    tiny_model.eval()
    with torch.no_grad():
        _, probas, _, _, _ = tiny_model(x, mask)
    assert (probas >= 0).all() and (probas <= 1).all()
