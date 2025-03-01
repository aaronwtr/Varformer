import torch
import math

import numpy as np

from torch import nn
from torch.nn import functional as F


class ShardedVarformer(nn.Module):
    def __init__(self, max_seq_len, num_muts, dropout, d_model, dim_feedforward, return_attn=False,
                 nhead=8, num_encoder_layers=2):
        super(ShardedVarformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.return_attn = return_attn

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
            num_layers=num_encoder_layers
        )

        self.pma = PMA(d_model=d_model, nhead=nhead, num_seeds=1, dropout=dropout)

    def forward(self, pathogenicity, position, mutation, mask):
        path_emb = self.pathogenicity_projection(pathogenicity.unsqueeze(-1))
        mut_emb = self.mutation_embedding(mutation.long())

        var_token = torch.cat([path_emb, mut_emb], dim=-1)
        var_token = self.positional_encoder(var_token, position)

        B, S, _ = var_token.size()

        var_token = var_token.permute(1, 0, 2)  # (B, S, E) -> (S, B, E) as required by nn.Transformer
        mask = mask.bool()

        output = self.variant_transformer(var_token, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)  # (S, B, E) -> (B, S, E)

        if self.return_attn:
            gene_embeddings, attn_weights = self.pma(output, return_attn_weights=True)
        else:
            gene_embeddings = self.pma(output, return_attn_weights=False)
            attn_weights = None

        return gene_embeddings, attn_weights


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.pe = self.generate_pe()

    def generate_pe(self):
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        periodicity = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(self.max_seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(positions * periodicity)
        pe[:, 1::2] = torch.cos(positions * periodicity)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)
        return pe

    def forward(self, x, positions):
        pe = self.pe.to(positions.device)
        pos_idx = positions.to(torch.long)
        pe = pe[pos_idx]
        x = x + pe
        return self.dropout(x)


class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention (Set Transformer style).
    This module expects shape [B, S, E] as input.
    """
    def __init__(self, d_model=512, nhead=8, num_seeds=1, dropout=0.1):
        """
        Args:
            d_model: dimension of each token embedding
            nhead: number of attention heads
            num_seeds: number of "learnable queries" for pooling
            dropout: dropout rate applied inside multi-head attention
        """
        super().__init__()
        self.num_seeds = num_seeds

        # Learnable seed (query) vectors, shape [num_seeds, d_model]
        self.seed_params = nn.Parameter(torch.randn(num_seeds, d_model))

        # Multi-head attention set up for [B, S, E] usage
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # even though transformer uses S,B,E, we want B,S,E here
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attn_weights=False):
        """
        Args:
            x: [B, S, d_model] – the tokens to pool
            return_attn_weights: if True, returns the attention matrix
                                 for interpretability

        Returns:
            out: [B, num_seeds, d_model]
            (optionally attn_weights): [B, num_seeds, S]
        """
        B, S, E = x.shape

        seeds = self.seed_params.unsqueeze(0).expand(B, -1, -1)  # [B, num_seeds, d_model]

        # Seeds attend to x
        #     query = seeds, key = x, value = x
        #     out shape: [B, num_seeds, d_model]
        #     attn_weights shape: [B, num_seeds, S]
        out, attn_weights = self.attn(
            query=seeds,
            key=x,
            value=x,
            need_weights=return_attn_weights
        )

        out = self.norm(out).squeeze(1)
        attn_weights = attn_weights.squeeze(1) if return_attn_weights else None

        return out, attn_weights  # shape [B, d_model], [B, S]
