"""Transformer encoder over per-gene variant sequences."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder that accepts explicit position indices.

    The PE tensor is stored as a plain attribute (not a registered buffer) so
    that it does not appear in state_dict; forward() re-casts it to the
    correct device on each call.
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
        pe = self.pe.to(positions.device)
        pos_idx = positions.to(torch.long)
        pe = pe[pos_idx]
        x = x + pe
        return self.dropout(x)


class VariantEncoder(nn.Module):
    """Transformer encoder over the variant sequence for a gene.

    Encodes each variant as the concatenation of a learned mutation embedding
    and a projected pathogenicity scalar, then applies a stack of transformer
    encoder layers conditioned on a sinusoidal positional encoding over the
    variant's protein position.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ``mutation_embedding_max_norm`` renormalises each embedding vector
        # whose L2 norm exceeds the cap.  ``None`` (inference default) is a plain
        # ``nn.Embedding`` for bit-exact parity with the published checkpoints;
        # a finite value (training default 20.0) prevents the embedding-norm
        # runaway that overflows the cross-attention layer under fp16.
        self.mutation_embedding = nn.Embedding(
            num_muts, d_model // 2, max_norm=mutation_embedding_max_norm
        )
        self.pathogenicity_projection = nn.Linear(1, d_model // 2)

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
        path_emb = self.pathogenicity_projection(pathogenicity.unsqueeze(-1))
        mut_emb = self.mutation_embedding(mutation.long())

        var_token = torch.cat([path_emb, mut_emb], dim=-1)
        var_token = self.positional_encoder(var_token, position)

        B, S, _ = var_token.size()

        # nn.TransformerEncoder expects (S, B, E)
        var_token = var_token.permute(1, 0, 2)
        mask = mask.bool()

        gene_embeddings = self.variant_transformer(var_token, src_key_padding_mask=mask)
        return gene_embeddings.permute(1, 0, 2)  # (B, S, E)
