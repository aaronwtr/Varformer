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

    def forward(self, pathogenicity, position, mutation, mask):
        path_emb = self.pathogenicity_projection(pathogenicity.unsqueeze(-1))
        mut_emb = self.mutation_embedding(mutation.long())

        var_token = torch.cat([path_emb, mut_emb], dim=-1)
        var_token = self.positional_encoder(var_token, position)

        B, S, _ = var_token.size()

        var_token = var_token.permute(1, 0, 2)  # (B, S, E) -> (S, B, E) as required by nn.Transformer
        mask = mask.bool()

        gene_embeddings = self.variant_transformer(var_token, src_key_padding_mask=mask)
        return gene_embeddings.permute(1, 0, 2)  # (S, B, E) -> (B, S, E)


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
