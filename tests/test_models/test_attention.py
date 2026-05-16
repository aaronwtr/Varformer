"""Tests for GeneVariantAttention."""
import torch
import pytest
from varformer.models.attention import GeneVariantAttention


@pytest.fixture
def tiny_gva():
    return GeneVariantAttention(
        gene_feature_dim=24,
        variant_feature_dim=32,
        attention_dim=32,
        nhead=1,
    )


def test_output_shapes(tiny_gva):
    B, S = 4, 10
    gene_feat = torch.rand(B, 24)
    var_emb = torch.rand(B, S, 32)

    attn_out, attn_w = tiny_gva(gene_feat, var_emb)
    assert attn_out.shape == (B, 32), f"Expected ({B},32), got {attn_out.shape}"
    assert attn_w.shape == (B, S), f"Expected ({B},{S}), got {attn_w.shape}"


def test_attn_weights_sum_to_one(tiny_gva):
    """Attention weights from softmax must sum to ~1 over variants."""
    B, S = 3, 8
    gene_feat = torch.rand(B, 24)
    var_emb = torch.rand(B, S, 32)

    _, attn_w = tiny_gva(gene_feat, var_emb)
    row_sums = attn_w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), f"Row sums: {row_sums}"


def test_attribute_names_for_checkpoint(tiny_gva):
    """query_layer, key_layer, value_layer, attn must exist as named children."""
    child_names = {name for name, _ in tiny_gva.named_children()}
    assert "query_layer" in child_names
    assert "key_layer" in child_names
    assert "value_layer" in child_names
    assert "attn" in child_names


def test_deterministic_in_eval(tiny_gva):
    tiny_gva.eval()
    B, S = 2, 6
    gene_feat = torch.rand(B, 24)
    var_emb = torch.rand(B, S, 32)

    with torch.no_grad():
        out1, w1 = tiny_gva(gene_feat, var_emb)
        out2, w2 = tiny_gva(gene_feat, var_emb)

    assert torch.allclose(out1, out2)
    assert torch.allclose(w1, w2)


def test_multi_head(tiny_gva):
    """Sanity-check with nhead>1 (divisor of attention_dim)."""
    gva_mh = GeneVariantAttention(
        gene_feature_dim=24,
        variant_feature_dim=32,
        attention_dim=32,
        nhead=4,
    )
    B, S = 2, 5
    gene_feat = torch.rand(B, 24)
    var_emb = torch.rand(B, S, 32)
    attn_out, attn_w = gva_mh(gene_feat, var_emb)
    assert attn_out.shape == (B, 32)
    assert attn_w.shape == (B, S)
