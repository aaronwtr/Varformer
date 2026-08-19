"""Tests for the VariantEncoder."""
import torch
import pytest
from varformer.models.variant_encoder import VariantEncoder, PositionalEncoder


@pytest.fixture
def tiny_ve():
    """Small VariantEncoder for fast CPU tests."""
    return VariantEncoder(
        max_seq_len=16,
        num_muts=50,
        dropout=0.0,
        d_model=32,
        dim_feedforward=64,
        nhead=4,
        num_encoder_layers=2,
    )


def test_output_shape(tiny_ve):
    B, S = 4, 10
    path = torch.rand(B, S)
    pos = torch.randint(0, 16, (B, S))
    mut = torch.randint(0, 50, (B, S))
    mask = torch.zeros(B, S, dtype=torch.bool)

    out = tiny_ve(path, pos, mut, mask)
    assert out.shape == (B, S, 32), f"Expected (4,10,32), got {out.shape}"


def test_mask_zeros_out_padding(tiny_ve):
    """Positions marked True in mask are padding; the output should differ
    depending on which positions are masked."""
    B, S = 2, 8
    path = torch.rand(B, S)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    mut = torch.zeros(B, S, dtype=torch.long)

    mask_no_pad = torch.zeros(B, S, dtype=torch.bool)
    mask_half_pad = torch.zeros(B, S, dtype=torch.bool)
    mask_half_pad[:, S // 2:] = True  # last half is padding

    out_no_pad = tiny_ve(path, pos, mut, mask_no_pad)
    out_half_pad = tiny_ve(path, pos, mut, mask_half_pad)

    # Unmasked positions should differ because self-attention context changes
    assert not torch.allclose(out_no_pad[:, :S // 2], out_half_pad[:, :S // 2])


def test_attribute_names_for_checkpoint(tiny_ve):
    """Attribute names must match the checkpoint state_dict keys."""
    state = dict(tiny_ve.named_parameters())
    assert any(k.startswith("mutation_embedding.") for k in state)
    assert any(k.startswith("pathogenicity_projection.") for k in state)
    assert any(k.startswith("positional_encoder.dropout.") or k.startswith("variant_transformer.") for k in state)


def test_positional_encoder_shape():
    pe = PositionalEncoder(max_seq_len=16, d_model=32, dropout=0.0)
    B, S = 3, 8
    x = torch.rand(B, S, 32)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    out = pe(x, pos)
    assert out.shape == (B, S, 32)


def test_deterministic_in_eval_mode(tiny_ve):
    """Dropout=0 and eval mode: output must be deterministic."""
    tiny_ve.eval()
    B, S = 2, 6
    path = torch.rand(B, S)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    mut = torch.randint(0, 50, (B, S))
    mask = torch.zeros(B, S, dtype=torch.bool)

    with torch.no_grad():
        out1 = tiny_ve(path, pos, mut, mask)
        out2 = tiny_ve(path, pos, mut, mask)

    assert torch.allclose(out1, out2)


def test_padding_mask_survives_autocast(tiny_ve):
    """Mixed precision must not trip the attn_mask dtype check.

    Handed a bool padding mask, nn.TransformerEncoder materialises a float mask
    in the pre-autocast dtype of ``src``; the query is then cast to bf16 and
    scaled_dot_product_attention rejects the mismatch with "Expected attn_mask
    dtype to be bool or to match query dtype". VariantEncoder builds the additive
    mask itself to keep the two dtypes consistent.
    """
    tiny_ve.eval()
    B, S = 3, 16
    pathogenicity = torch.rand(B, S)
    position = torch.randint(0, S, (B, S))
    mutation = torch.randint(0, 50, (B, S))
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, 10:] = True

    with torch.no_grad():
        reference = tiny_ve(pathogenicity, position, mutation, mask)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_out = tiny_ve(pathogenicity, position, mutation, mask)

    assert autocast_out.shape == reference.shape
    assert torch.isfinite(autocast_out).all()
    # bf16 has ~3 decimal digits; this is a dtype-compatibility check, not a
    # numerical-equivalence one.
    assert (reference - autocast_out.float()).abs().max() < 0.1


def test_padded_positions_do_not_leak(tiny_ve):
    """Changing the contents of masked positions must not move real ones."""
    tiny_ve.eval()
    B, S = 3, 16
    pathogenicity = torch.rand(B, S)
    position = torch.randint(0, S, (B, S))
    mutation = torch.randint(0, 50, (B, S))
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, 10:] = True

    perturbed = mutation.clone()
    perturbed[:, 10:] = (perturbed[:, 10:] + 7) % 50

    with torch.no_grad():
        original = tiny_ve(pathogenicity, position, mutation, mask)
        changed = tiny_ve(pathogenicity, position, perturbed, mask)

    assert torch.equal(original[:, :10], changed[:, :10])
