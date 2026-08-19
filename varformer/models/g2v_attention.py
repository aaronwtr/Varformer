"""Gene-to-variant cross-attention layer."""
from __future__ import annotations

from typing import Optional

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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gene-variant cross-attention.

        Args:
            gene_features:       [B, gene_feature_dim] — combined gene context.
            variant_embeddings:  [B, S, variant_feature_dim] — output from VariantEncoder.
            key_padding_mask:     [B, S] bool; ``True`` marks padded variants.
                A fully padded row produces a zero representation and zero
                attention weights rather than NaNs.

        Returns:
            attn_output:  [B, attention_dim] — variant-informed gene representation.
            attn_weights: [B, S] — attention scores over variants.
        """
        B, S, E = variant_embeddings.shape

        Q = self.query_layer(gene_features).unsqueeze(1)  # [B, 1, attention_dim]
        K = self.key_layer(variant_embeddings)            # [B, S, attention_dim]
        V = self.value_layer(variant_embeddings)          # [B, S, attention_dim]

        # MultiheadAttention softmaxes an all-masked row of ``-inf`` values,
        # which yields NaNs. Temporarily expose one key for those rows so the
        # operation remains finite, then explicitly return the neutral result.
        fully_padded = None
        safe_key_padding_mask = key_padding_mask
        if key_padding_mask is not None:
            safe_key_padding_mask = key_padding_mask.to(device=K.device, dtype=torch.bool)
            fully_padded = safe_key_padding_mask.all(dim=1)
            if fully_padded.any():
                safe_key_padding_mask = safe_key_padding_mask.clone()
                safe_key_padding_mask[fully_padded, 0] = False

        attn_output, attn_weights = self.attn(
            Q,
            K,
            V,
            key_padding_mask=safe_key_padding_mask,
        )

        attn_output = attn_output.squeeze(1)   # [B, attention_dim]
        attn_weights = attn_weights.squeeze(1)  # [B, S]

        if fully_padded is not None and fully_padded.any():
            attn_output = attn_output.masked_fill(fully_padded.unsqueeze(1), 0.0)
            attn_weights = attn_weights.masked_fill(fully_padded.unsqueeze(1), 0.0)

        return attn_output, attn_weights
