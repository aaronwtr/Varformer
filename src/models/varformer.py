import torch
from torch import nn


class VariantEmbedding(nn.Module):
    def __init__(self, num_mutations, max_seq_length, num_genes, d_model):
        super(VariantEmbedding, self).__init__()
        self.pathogenicity_embed = nn.Linear(1, d_model // 4)
        self.position_embed = nn.Embedding(max_seq_length + 1, d_model // 4)
        self.mutation_embed = nn.Embedding(num_mutations + 1, d_model // 4)
        self.gene_embed = nn.Embedding(num_genes + 1, d_model // 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, pathogenicity, position, mutation, gene):
        path_emb = self.pathogenicity_embed(pathogenicity.unsqueeze(-1))
        pos_emb = self.position_embed(position.to(torch.int))
        mut_emb = self.mutation_embed(mutation.to(torch.int))
        gene_emb = self.gene_embed(gene.to(torch.int))
        combined = torch.cat([path_emb, pos_emb, mut_emb, gene_emb], dim=-1)
        combined = self.layer_norm(combined)
        batch_size = combined.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, combined], dim=1)


class ShardAttention(nn.Module):
    """
    The ShardAttention module performs multi-head attention over the different shards of gene and constructs a final
    gene-level embedding for each gene in the batch. Note that in order to perform attention across the shards,
    the batches (i.e. number of shards) are considered to be the sequence length, and the batch size is set to 1.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return self.norm(query + attn_output)


class GeneAggregator(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.attention = ShardAttention(d_model, nhead)

    def forward(self, shard_embeds):
        if len(shard_embeds.shape) == 2:
            shard_embeds = shard_embeds.unsqueeze(1)
        return self.attention(shard_embeds, shard_embeds, shard_embeds)


class Varformer(nn.Module):
    def __init__(self, num_mutations, max_seq_length, num_genes, d_model, nhead, num_layers):
        super(Varformer, self).__init__()

        dim_feedforward = int(d_model // 2)
        self.variant_embedding = VariantEmbedding(num_mutations, max_seq_length, num_genes, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, pathogenicity, position, mutation, gene, src_mask):
        # Adjust mask for <CLS> token
        cls_mask = torch.ones((src_mask.size(0), 1), dtype=torch.bool, device=src_mask.device)
        src_mask = torch.cat([cls_mask, src_mask], dim=1)

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # Use the <CLS> token representation for classification
        cls_output = output[:, 0, :]  # First token is <CLS>
        return cls_output
        # return torch.sigmoid(self.classifier(cls_output))


class ShardedVarformer(nn.Module):
    def __init__(self, max_seq_len, num_muts, num_genes, dropout, d_model=128, nhead=2, num_encoder_layers=2):
        super(ShardedVarformer, self).__init__()
        self.num_genes = num_genes
        self.pathogenicity_embed = nn.Linear(1, d_model // 4)
        self.position_embed = nn.Embedding(max_seq_len + 1, d_model // 4)
        self.mutation_embed = nn.Embedding(num_muts + 1, d_model // 4)
        self.gene_embed = nn.Embedding(num_genes + 1, d_model // 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.variant_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_encoder_layers
        )

        self.shard_aggregation = nn.Linear(d_model, d_model)

    def forward(self, pathogenicity, position, mutation, gene, mask):
        pat_embed = self.pathogenicity_embed(pathogenicity.unsqueeze(-1))
        pos_embed = self.position_embed(position)
        mut_embed = self.mutation_embed(mutation)
        gene_embed = self.gene_embed(gene)

        x = torch.cat([pat_embed, pos_embed, mut_embed, gene_embed], dim=-1)

        x = self.layer_norm(x)
        x = self.dropout(x)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x.permute(1, 0, 2)  # (B, S, E) --> (S, B, E) as default required by transformer
        mask = mask.bool()

        # Adjust mask for <CLS> token
        cls_mask = torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Note: we invert the mask to indicate which elements should be masked (True)
        output = self.variant_transformer(x, src_key_padding_mask=~mask)

        # Use the <CLS> token representation for classification
        cls_output = output[0, :, :]
        return cls_output
