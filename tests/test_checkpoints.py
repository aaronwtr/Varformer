import re
import torch
from pathlib import Path
from varformer.checkpoints import _remap_legacy_state_dict, find_checkpoint, best_seed


def test_strip_metric_keys():
    state_dict = {
        "model.gc_projection.0.weight": torch.zeros(8, 30),
        "model.acc.threshold": torch.tensor(0.5),
        "model.auroc.thresholds": torch.tensor([0.0, 0.5, 1.0]),
        "model.recall.threshold": torch.tensor(0.5),
        "model.precision.threshold": torch.tensor(0.5),
        "model.f1.threshold": torch.tensor(0.5),
        "model.auprc.thresholds": torch.tensor([0.0]),
        "model.spearman.preds": torch.zeros(10),
    }
    out = _remap_legacy_state_dict(state_dict)
    assert "model.gc_projection.0.weight" in out
    assert not any(k.startswith("model.acc.") for k in out)
    assert not any(k.startswith("model.auroc.") for k in out)
    assert not any(k.startswith("model.recall.") for k in out)
    assert not any(k.startswith("model.precision.") for k in out)
    assert not any(k.startswith("model.f1.") for k in out)
    assert not any(k.startswith("model.auprc.") for k in out)
    assert not any(k.startswith("model.spearman.") for k in out)


def test_strip_dead_classifier():
    state_dict = {
        "model.varformer.classifier.weight": torch.zeros(1, 256),
        "model.varformer.classifier.bias": torch.zeros(1),
        "model.varformer.varformer.variant_transformer.layers.0.norm1.weight": torch.zeros(256),
    }
    out = _remap_legacy_state_dict(state_dict)
    assert "model.varformer.classifier.weight" not in out
    assert "model.varformer.classifier.bias" not in out
    # The wrapper-renamed key should still be there
    assert "model.varformer.variant_transformer.layers.0.norm1.weight" in out


def test_collapse_wrapper_prefix():
    state_dict = {
        "model.varformer.varformer.variant_transformer.layers.0.norm1.weight": torch.zeros(256),
        "model.varformer.varformer.mutation_embedding.weight": torch.zeros(400, 128),
        "model.varformer.varformer.positional_encoder.pe": torch.zeros(1024, 256),
    }
    out = _remap_legacy_state_dict(state_dict)
    assert "model.varformer.variant_transformer.layers.0.norm1.weight" in out
    assert "model.varformer.mutation_embedding.weight" in out
    assert "model.varformer.positional_encoder.pe" in out
    # Original wrapper-prefixed keys are gone
    assert not any(".varformer.varformer." in k for k in out)


def test_does_not_touch_unrelated_keys():
    state_dict = {
        "model.gc_projection.0.weight": torch.zeros(8, 30),
        "model.gc_projection.0.bias": torch.zeros(8),
        "model.go_projection.1.weight": torch.zeros(16),
        "model.gene_variant_attention.query_layer.weight": torch.zeros(32, 24),
        "model.classification_head.0.weight": torch.zeros(28, 56),
    }
    out = _remap_legacy_state_dict(state_dict)
    assert out == state_dict  # unchanged


def test_real_checkpoint_remap():
    """Sanity-check on a real published checkpoint: cleaning must produce a non-empty state_dict
    with no metric or wrapper keys remaining."""
    ckpt_dir = Path(__file__).resolve().parents[1] / "src" / "checkpoints" / "nfe"
    cands = list(ckpt_dir.glob("seed42-epoch=*-val_spearman=*.ckpt"))
    if not cands:
        import pytest
        pytest.skip("Real checkpoint not available locally")
    raw = torch.load(cands[0], map_location="cpu")
    cleaned = _remap_legacy_state_dict(raw["state_dict"])
    assert len(cleaned) > 0
    assert not any(re.match(r"^model\.(acc|auroc|recall|precision|f1|auprc|spearman)\.", k) for k in cleaned)
    assert not any(k.startswith("model.varformer.classifier.") for k in cleaned)
    assert not any(".varformer.varformer." in k for k in cleaned)
    # Sanity: known-good keys should still be present
    assert any(k.startswith("model.gc_projection.") for k in cleaned)
    assert any(k.startswith("model.varformer.variant_transformer.") for k in cleaned)
