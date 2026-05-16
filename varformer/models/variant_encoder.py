"""VariantEncoder: transformer over per-gene variant sequences.

Renamed from ShardedVarformer (src/models/varformer.py). Internal attribute names
are preserved verbatim so legacy checkpoint state_dict keys continue to match.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder that accepts explicit position indices.

    The PE tensor is not a registered buffer: it is generated on CPU/GPU at
    construction time, stored as a plain attribute, and re-cast to the input
    device inside forward(). This matches the original src/ behaviour exactly.
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)
        return pe

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        pe = self.pe.to(positions.device)
        pos_idx = positions.to(torch.long)
        pe = pe[pos_idx]
        x = x + pe
        return self.dropout(x)


class VariantEncoder(nn.Module):
    """Transformer encoder over the variant sequence for a gene.

    Renamed from ShardedVarformer. All attribute names (mutation_embedding,
    pathogenicity_projection, positional_encoder, dropout, variant_transformer)
    are identical to the originals so checkpoint keys load without remapping.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.mutation_embedding = nn.Embedding(num_muts, d_model // 2)
        self.pathogenicity_projection = nn.Linear(1, d_model // 2)

        self.positional_encoder = PositionalEncoder(max_seq_len, d_model, dropout)
        self.dropout = nn.Dropout(dropout)

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
