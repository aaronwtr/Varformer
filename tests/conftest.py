"""Shared pytest fixtures. Tiny synthetic data — CPU-only.

Fixtures are progressively filled in by Phases 3, 4, 5, 7.
"""
import pytest


@pytest.fixture
def tiny_hyperparams():
    """Minimal hyperparameters for fast unit tests."""
    return {
        "d_model": 32,
        "max_seq_len": 16,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "gc_width": 8,
        "go_width": 16,
        "gv_attn_dim": 32,
        "depth_cls_head": 2,
        "threshold": 0.5,
        "return_attn": True,
        "use_pvc": True,
        "precision": "32",
    }
