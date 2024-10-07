import torch
from torch import nn
import numpy as np
import math


class ShardedVarformer(nn.Module):
    # TODO:
    #  [ ] Train model with best settings for as long as possible.
    #  [ ] Ablate gene tokens and evaluate
    #  [ ] Look into the advantage of <PAD> tokens over 0 padding
    #  [ ] Add Varformer embeddings to the GO and GC features and evaluate.
    #  [ ] Come up with test cases to compare inclusion of Varformer embeddings in the GO and GC features.
    def __init__(self, max_seq_len, num_muts, dropout, d_model, nhead=2, num_encoder_layers=2):
        super(ShardedVarformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Tokenization
        self.mutation_embedding = nn.Embedding(num_muts, d_model // 2)
        self.pathogenicity_projection = nn.Linear(1, d_model // 2)
        self.shared_representation = nn.Linear(d_model, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.variant_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2 * d_model,
                activation="gelu"),
            num_layers=num_encoder_layers
        )

        # Pooling layer
        self.pooling = nn.Linear(d_model, d_model)

        # Positional encoding
        self.positional_encoder = PositionalEncoder(max_seq_len, d_model // 2, dropout)

    def forward(self, pathogenicity, position, mutation, mask):
        return self.get_embeddings(pathogenicity, position, mutation, mask)

    def get_embeddings(self, pathogenicity, position, mutation, mask):
        path_tokens = self.pathogenicity_projection(pathogenicity.unsqueeze(-1))
        path_tokens_pe = self.positional_encoder(path_tokens, position)
        mut_tokens = self.mutation_embedding(mutation)
        mut_tokens_pe = self.positional_encoder(mut_tokens, position)

        x = torch.cat([path_tokens_pe, mut_tokens_pe], dim=-1)
        x = self.shared_representation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x.permute(1, 0, 2)  # (B, S, E) -> (S, B, E) as required by nn.Transformer
        mask = mask.bool()
        cls_mask = torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        output = self.variant_transformer(x, src_key_padding_mask=~mask)
        output = output.permute(1, 0, 2)    # (S, B, E) -> (B, S, E)

        cls_output = output[:, 0]
        pooled_output = self.pooling(output[:, 1:]).mean(dim=1)

        gene_embedding = (pooled_output + cls_output) / 2
        gene_embedding = self.dropout(gene_embedding)

        return gene_embedding


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
        pe = pe[positions]
        x = x + pe
        return self.dropout(x)
