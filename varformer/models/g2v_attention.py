"""Gene-to-variant cross-attention layer."""
from __future__ import annotations

import torch
from torch import nn


class GeneVariantAttention(nn.Module):
    """Cross-attention mechanism: gene features attend over variant embeddings.

    Args:
        gene_feature_dim:    Dimensionality of combined gene context (GC + GO).
        variant_feature_dim: Embedding size of variant features (output of VariantEncoder).
        attention_dim:       Projection dimension for the attention space.
        nhead:               Number of attention heads (default 1 for interpretability).
    """

    def __init__(
        self,
        gene_feature_dim: int,
        variant_feature_dim: int,
        attention_dim: int,
        nhead: int = 1,
    ):
        super().__init__()

        self.query_layer = nn.Linear(gene_feature_dim, attention_dim)   # Gene as Query
        self.key_layer = nn.Linear(variant_feature_dim, attention_dim)  # Variant as Key
        self.value_layer = nn.Linear(variant_feature_dim, attention_dim)  # Variant as Value

        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=nhead, batch_first=True)

    def forward(
        self,
        gene_features: torch.Tensor,
        variant_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gene-variant cross-attention.

        Args:
            gene_features:       [B, gene_feature_dim] — combined gene context.
            variant_embeddings:  [B, S, variant_feature_dim] — output from VariantEncoder.

        Returns:
            attn_output:  [B, attention_dim] — variant-informed gene representation.
            attn_weights: [B, S] — attention scores over variants.
        """
        B, S, E = variant_embeddings.shape

        Q = self.query_layer(gene_features).unsqueeze(1)  # [B, 1, attention_dim]
        K = self.key_layer(variant_embeddings)            # [B, S, attention_dim]
        V = self.value_layer(variant_embeddings)          # [B, S, attention_dim]

        attn_output, attn_weights = self.attn(Q, K, V)

        attn_output = attn_output.squeeze(1)   # [B, attention_dim]
        attn_weights = attn_weights.squeeze(1)  # [B, S]

        return attn_output, attn_weights
