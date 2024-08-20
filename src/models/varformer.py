import torch
from torch import nn


class VariantEmbedding(nn.Module):
    def __init__(self, num_mutations, max_seq_length, num_genes, d_model):
        super(VariantEmbedding, self).__init__()
        self.pathogenicity_embed = nn.Linear(1, d_model // 4)
        self.position_embed = nn.Embedding(max_seq_length, d_model // 4)
        self.mutation_embed = nn.Embedding(num_mutations, d_model // 4)
        self.gene_embed = nn.Embedding(num_genes, d_model // 4)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, pathogenicity, position, mutation, gene):
        path_emb = self.pathogenicity_embed(pathogenicity.unsqueeze(-1))
        pos_emb = self.position_embed(position.to(torch.int))
        mut_emb = self.mutation_embed(mutation.to(torch.int))
        gene_emb = self.gene_embed(gene.to(torch.int))
        combined = torch.cat([path_emb, pos_emb, mut_emb, gene_emb], dim=-1)
        return self.layer_norm(combined)


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
        src = self.variant_embedding(pathogenicity, position, mutation, gene)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = output.mean(dim=1)  # Global average pooling
        return torch.sigmoid(self.classifier(output))


class ShardedVarformer(nn.Module):
    def __init__(self, max_seq_len, num_muts, num_genes, d_model=128, nhead=2, num_encoder_layers=2):
        super(ShardedVarformer, self).__init__()
        self.num_genes = num_genes
        self.pathogenicity_embed = nn.Linear(1, d_model // 4)
        self.position_embed = nn.Embedding(max_seq_len + 1, d_model // 4)
        self.mutation_embed = nn.Embedding(num_muts + 1, d_model // 4)
        self.gene_embed = nn.Embedding(num_genes, d_model // 4)

        self.layer_norm = nn.LayerNorm(d_model)

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

        x = x.permute(1, 0, 2)  # (B, S, E) --> (S, B, E) as default required by transformer
        mask = mask.bool()

        # Note: we invert the mask to indicate which elements should be masked (True)
        output = self.variant_transformer(x, src_key_padding_mask=~mask)

        # Create shard-level embedding by averaging over non-padded elements
        shard_output = (output.permute(1, 0, 2) * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        return shard_output
