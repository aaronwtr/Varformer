"""Transformer encoder over per-gene variant sequences."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder indexed by explicit protein positions.

    ``forward(x, positions)`` adds the sinusoidal encoding for each variant's
    protein position to its embedding, then applies dropout.
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.pe = self.generate_pe()

    def generate_pe(self) -> torch.Tensor:
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        periodicity = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(self.max_seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(positions * periodicity)
        pe[:, 1::2] = torch.cos(positions * periodicity)
        return pe

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        pos_idx = positions.to(torch.long)
        if torch.any(pos_idx < 0):
            raise ValueError("Protein positions must be non-negative.")

        # Preserve the original lookup path for modulo inputs while allowing
        # collision-free absolute protein coordinates beyond max_seq_len.
        if pos_idx.numel() == 0 or int(pos_idx.max()) < self.max_seq_len:
            pe = self.pe.to(device=positions.device, dtype=x.dtype)[pos_idx]
        else:
            pos = pos_idx.to(device=positions.device, dtype=torch.float32).unsqueeze(-1)
            periodicity = torch.exp(
                torch.arange(0, self.d_model, 2, device=positions.device, dtype=torch.float32)
                * (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(*pos_idx.shape, self.d_model, device=positions.device)
            pe[..., 0::2] = torch.sin(pos * periodicity)
            pe[..., 1::2] = torch.cos(pos * periodicity)
            pe = pe.to(dtype=x.dtype)
        x = x + pe
        return self.dropout(x)


def _autocast_dtype(reference: torch.Tensor) -> torch.dtype:
    """The dtype attention will compute in for ``reference``.

    Returns the active autocast dtype when autocast is enabled for the tensor's
    device, otherwise the tensor's own dtype. torch 2.1 exposes this per device
    rather than through a single ``get_autocast_dtype``.
    """
    if reference.is_cuda and torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    if not reference.is_cuda and torch.is_autocast_cpu_enabled():
        return torch.get_autocast_cpu_dtype()
    return reference.dtype


class VariantEncoder(nn.Module):
    """Transformer encoder over the variant sequence for a gene.

    Encodes each variant as the concatenation of a learned mutation embedding
    and a learned pathogenicity embedding (a representation derived from the
    variant's continuous pathogenicity score), then applies a stack of
    transformer encoder layers conditioned on a sinusoidal positional encoding
    over the variant's protein position.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_muts: int,
        dropout: float,
        d_model: int,
        dim_feedforward: int,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        mutation_embedding_max_norm: float = None,
        variant_token_schema: str = "original",
        token_ablate: tuple = (),
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.variant_token_schema = variant_token_schema
        self.token_ablate = tuple(token_ablate)

        # mutation_embedding_max_norm caps each embedding vector's L2 norm;
        # None leaves the embedding unconstrained.
        self.mutation_embedding = nn.Embedding(
            num_muts, d_model // 2, max_norm=mutation_embedding_max_norm
        )
        continuous_dimensions = {
            "original": 1,
            "af": 2,
            "af_relative_position": 3,
            "af_relative_position_confidence": 4,
        }
        if variant_token_schema not in continuous_dimensions:
            raise ValueError(f"Unknown variant token schema: {variant_token_schema}")
        self.pathogenicity_projection = nn.Linear(
            continuous_dimensions[variant_token_schema], d_model // 2
        )

        self.positional_encoder = PositionalEncoder(max_seq_len, d_model, dropout)

        self.variant_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=num_encoder_layers,
        )

    def forward(
        self,
        pathogenicity: torch.Tensor,
        position: torch.Tensor,
        mutation: torch.Tensor,
        mask: torch.Tensor,
        allele_frequency: torch.Tensor = None,
        relative_position: torch.Tensor = None,
        frequency_confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode a batch of variant sequences.

        Args:
            pathogenicity: [B, S] AlphaMissense scores.
            position:      [B, S] integer protein positions (0-indexed).
            mutation:      [B, S] integer mutation indices.
            mask:          [B, S] bool; True where positions are padding.

        Returns:
            Tensor [B, S, d_model] — contextualised variant embeddings.
        """
        continuous = [pathogenicity]
        if self.variant_token_schema != "original":
            if allele_frequency is None:
                raise ValueError(f"{self.variant_token_schema} requires allele_frequency")
            continuous.append(allele_frequency)
        if self.variant_token_schema in {
            "af_relative_position", "af_relative_position_confidence"
        }:
            if relative_position is None:
                raise ValueError(f"{self.variant_token_schema} requires relative_position")
            continuous.append(relative_position)
        if self.variant_token_schema == "af_relative_position_confidence":
            if frequency_confidence is None:
                raise ValueError("af_relative_position_confidence requires frequency_confidence")
            continuous.append(frequency_confidence)
        path_emb = self.pathogenicity_projection(torch.stack(continuous, dim=-1))
        mut_emb = self.mutation_embedding(mutation.long())
        if "mutation" in self.token_ablate:
            mut_emb = torch.zeros_like(mut_emb)

        var_token = torch.cat([path_emb, mut_emb], dim=-1)
        # The enhanced relative-position schemas carry q/L directly as a
        # continuous token feature. They deliberately do not also add the
        # collision-prone q modulo 1,024 positional encoding.
        if self.variant_token_schema in {"original", "af"} and "position" not in self.token_ablate:
            var_token = self.positional_encoder(var_token, position)
        else:
            var_token = self.positional_encoder.dropout(var_token)

        B, S, _ = var_token.size()

        # nn.TransformerEncoder expects (S, B, E)
        var_token = var_token.permute(1, 0, 2)
        mask = mask.bool()

        # Build the additive padding mask ourselves rather than handing
        # nn.TransformerEncoder a bool one. Given a bool mask it calls
        # F._canonical_mask, which materialises a float mask in the dtype of
        # ``src`` *before* autocast has cast anything. Under mixed precision the
        # query is then bf16/fp16 while the mask is still fp32, and
        # scaled_dot_product_attention rejects the pair with
        # "Expected attn_mask dtype to be bool or to match query dtype".
        # Constructing it in the dtype attention will actually compute in keeps
        # the two consistent, and is a no-op outside autocast.
        padding_mask = torch.zeros_like(
            mask, dtype=_autocast_dtype(var_token)
        ).masked_fill_(mask, float("-inf"))

        gene_embeddings = self.variant_transformer(
            var_token, src_key_padding_mask=padding_mask
        )
        return gene_embeddings.permute(1, 0, 2)  # (B, S, E)
