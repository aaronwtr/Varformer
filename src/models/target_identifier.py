import torch

import torch.nn as nn

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, Recall, Precision, F1Score, AveragePrecision
from models.varformer import ShardedVarformer


class BaseTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features):
        super(BaseTargetIdentifier, self).__init__()
        self.config = config['hyperparameters']

        # **REMOVED Feature extraction layers (MLP)**
        # feature_layers = []
        # layer_sizes = [num_features] + [int(self.config['width'])] * int(self.config['num_layers'])
        # layer_size_prev = layer_sizes[0]
        # for layer_size in layer_sizes[1:]:
        #     feature_layers += [
        #         torch.nn.Linear(layer_size_prev, layer_size),
        #         torch.nn.BatchNorm1d(layer_size),
        #         torch.nn.ReLU(),
        #         torch.nn.Dropout(p=float(self.config['dropout']))
        #     ]
        #     layer_size_prev = layer_size
        # self.feature_extractor = torch.nn.Sequential(*feature_layers)

        # Store final classification layer separately
        # **MODIFIED classifier input dimension to be directly from input features**
        self.classifier = torch.nn.Linear(num_features, 1)  # Input dimension is now num_features, not layer_sizes[-1]

        # **REMOVED full sequential model**
        # self.layers = torch.nn.Sequential(
        #     self.feature_extractor,
        #     self.classifier
        # )

        self.init_weights = self.initialise_weights()

        self.acc = Accuracy(task="binary", threshold=config['hyperparameters']['threshold'])
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=config['hyperparameters']['threshold'])
        self.precision = Precision(task="binary", threshold=config['hyperparameters']['threshold'])
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary", threshold=config['hyperparameters']['threshold'])
        self.spearman = SpearmanCorrCoef()

    def forward(self, x, return_features=False):
        # **REMOVED Feature extraction step - input x is now directly features**
        # features = self.feature_extractor(x)
        features = x  # Input x is now directly the features for the classifier

        # Get logits
        logits = self.classifier(features).squeeze()

        # Calculate probabilities and predictions
        probabilities = torch.sigmoid(logits)  # Using torch.sigmoid directly
        binary_predictions = (probabilities > float(self.config['threshold'])).float()

        if return_features:
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': binary_predictions,
                'features': features
            }

        return logits, probabilities, binary_predictions

    def initialise_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        initial_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                initial_weights[name + ".weight"] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    initial_weights[name + ".bias"] = module.bias.detach().clone()
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                initial_weights[name + ".weight"] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
                    initial_weights[name + ".bias"] = module.bias.detach().clone()
                module.running_mean.fill_(0)
                module.running_var.fill_(1)
                initial_weights[name + ".running_mean"] = module.running_mean.detach().clone()
                initial_weights[name + ".running_var"] = module.running_var.detach().clone()
        return initial_weights


class VarformerTargetIdentifier(BaseTargetIdentifier):
    def __init__(self, config, d_model, num_mutations, max_seq_len):
        super(VarformerTargetIdentifier, self).__init__(config, d_model)

        varformer_config = config['hyperparameters']
        self.varformer = ShardedVarformer(
            max_seq_len=max_seq_len,
            num_muts=num_mutations,
            dropout=float(varformer_config['dropout']),
            d_model=varformer_config['d_model'],
            dim_feedforward=varformer_config['dim_feedforward'],
            return_attn=varformer_config['return_attn'],
            nhead=varformer_config['nhead'],
            num_encoder_layers=varformer_config['num_encoder_layers']
        )

    def forward(self, x, mask=None):
        return self.varformer(x['pathogenicity'], x['position'], x['mutation'], mask)


class MultiModalTargetIdentifierV1(BaseTargetIdentifier):
    def __init__(self, config, num_features_gc, num_features_go, num_mutations, max_seq_len,
                 num_genes):
        super(MultiModalTargetIdentifierV1, self).__init__(config, num_features_gc + num_features_go + max_seq_len)

        self.num_features_gc = num_features_gc
        self.num_features_go = num_features_go
        self.num_mutations = num_mutations
        self.max_seq_len = max_seq_len
        self.num_genes = num_genes

        self.gc_branch = BaseTargetIdentifier(
            config=config,
            num_features=num_features_gc
        )

        self.go_branch = BaseTargetIdentifier(
            config=config,
            num_features=num_features_go
        )

        self.pvc_branch = VarformerTargetIdentifier(
            config=config,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            d_model=int(config['hyperparameters']['d_model'])
        )

        # Create the classification branch
        combined_input_size = int(self.config['width']) * 2 + int(self.config['d_model'])
        bottleneck_size = combined_input_size // 4

        classification_layers = [
            torch.nn.Linear(combined_input_size, int(self.config['width'])),
            torch.nn.BatchNorm1d(int(self.config['width'])),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=float(self.config['dropout'])),

            torch.nn.Linear(int(self.config['width']), bottleneck_size),
            torch.nn.BatchNorm1d(bottleneck_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=float(self.config['dropout'])),

            torch.nn.Linear(bottleneck_size, 1)
        ]

        self.classification_branch = torch.nn.Sequential(*classification_layers)

        self.init_weights = self.initialise_weights()

    def forward(self, x, mask=None):
        gc_output_dict = self.gc_branch(x['gc'][0], return_features=True)
        gc_features = gc_output_dict['features']

        go_output_dict = self.go_branch(x['go'][0], return_features=True)
        go_features = go_output_dict['features']

        gene_var_emb, attn_weights = self.pvc_branch(
            {
                'pathogenicity': x['pvc']['pathogenicity'],
                'position': x['pvc']['position'],
                'mutation': x['pvc']['mutation']
            },
            mask=mask
        )

        combined_features = torch.cat([gc_features, go_features, gene_var_emb], dim=-1)

        logits = self.classification_branch(combined_features).squeeze()
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()

        return logits, probabilities, binary_predictions


class MultiModalTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features_gc, num_features_go, num_mutations, max_seq_len, num_genes):
        super(MultiModalTargetIdentifier, self).__init__()

        self.num_features_gc = num_features_gc
        self.num_features_go = num_features_go
        self.num_mutations = num_mutations
        self.max_seq_len = max_seq_len
        self.num_genes = num_genes
        self.config = config
        self.hyperparams = self.config['hyperparameters']
        self.dropout = nn.Dropout(float(self.hyperparams['dropout']))

        # 1. GC Branch MLP
        gc_width = int(config['hyperparameters']['gc_width'])
        # gc_layers = []
        # layer_sizes_gc = [num_features_gc] + [gc_width] * num_layers  # MLP layer sizes for GC branch
        # layer_size_prev_gc = layer_sizes_gc[0]
        # for layer_size_gc in layer_sizes_gc[1:]:
        #     gc_layers += [
        #         nn.Linear(layer_size_prev_gc, layer_size_gc),
        #         nn.BatchNorm1d(layer_size_gc),
        #         nn.ReLU(),
        #         nn.Dropout(p=float(config['hyperparameters']['dropout']))
        #     ]
        #     layer_size_prev_gc = layer_size_gc
        # self.gc_feature_extractor = nn.Sequential(*gc_layers)  # Sequential MLP for GC branch

        self.gc_projection = nn.Sequential(
            nn.Linear(num_features_gc, gc_width),
            nn.LayerNorm(gc_width),
            nn.ReLU(),
            nn.Dropout(p=float(config['hyperparameters']['dropout']))
        )

        # 2. GO Branch MLP
        # go_layers = []
        # go_width = int(config['hyperparameters']['go_width'])
        # layer_sizes_go = [num_features_go] + [go_width] * num_layers  # MLP layer sizes for GO branch
        # layer_size_prev_go = layer_sizes_go[0]
        # for layer_size_go in layer_sizes_go[1:]:
        #     go_layers += [
        #         nn.Linear(layer_size_prev_go, layer_size_go),
        #         nn.BatchNorm1d(layer_size_go),
        #         nn.ReLU(),
        #         nn.Dropout(p=float(config['hyperparameters']['dropout']))
        #     ]
        #     layer_size_prev_go = layer_size_go
        # self.go_feature_extractor = nn.Sequential(*go_layers)  # Sequential MLP for GO branch

        go_width = int(config['hyperparameters']['go_width'])
        self.go_projection = nn.Sequential(
            nn.Linear(num_features_go, go_width),
            nn.LayerNorm(go_width),
            nn.ReLU(),
            nn.Dropout(p=float(config['hyperparameters']['dropout']))
        )

        # 3. PVC Branch (Transformer)
        self.varformer = VarformerTargetIdentifier(
            config=config,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            d_model=int(config['hyperparameters']['d_model']),
        )

        # Gene-Variant Attention
        gene_feature_dim = gc_width + go_width
        variant_feature_dim = int(config['hyperparameters']['d_model'])
        attention_dim = int(config['hyperparameters']['gv_attn_dim'])

        self.gene_variant_attention = GeneVariantAttention(
            gene_feature_dim=gene_feature_dim,
            variant_feature_dim=variant_feature_dim,
            attention_dim=attention_dim,
            nhead=int(config['hyperparameters']['nhead'])
        )

        # num_layers = int(config['hyperparameters']['num_layers'])
        # inp_dim_classifier = gene_feature_dim + attention_dim
        # self.classification_head = nn.Sequential(
        #     nn.Linear(inp_dim_classifier, inp_dim_classifier // 2),
        #     nn.ReLU(),
        #     self.dropout,
        #     nn.Linear(inp_dim_classifier // 2, inp_dim_classifier // 4),
        #     nn.ReLU(),
        #     self.dropout,
        #     nn.Linear(inp_dim_classifier // 4, 1)
        # )

        cls_head_layers = []
        depth_cls_head = int(config['hyperparameters']['depth_cls_head'])
        inp_dim_classifier = gene_feature_dim + attention_dim
        current_dim = inp_dim_classifier
        for _ in range(depth_cls_head):
            hidden_dim = current_dim // 2
            cls_head_layers += [
                nn.Linear(current_dim, hidden_dim),  # Use current_dim instead of inp_dim_classifier
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=float(config['hyperparameters']['dropout']))
            ]
            current_dim = hidden_dim  # Update current_dim for the next layer
        cls_head_layers += [nn.Linear(current_dim, 1)]
        self.classification_head = nn.Sequential(*cls_head_layers)

        self.acc = Accuracy(task="binary", threshold=self.hyperparams['threshold'])
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=self.hyperparams['threshold'])
        self.precision = Precision(task="binary", threshold=self.hyperparams['threshold'])
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary", threshold=self.hyperparams['threshold'])
        self.spearman = SpearmanCorrCoef()

    def forward(self, x, mask=None):
        # 1. Process GC and GO features through their linear layers
        z_gc = self.gc_projection(x['gc'][0])
        z_go = self.go_projection(x['go'][0])
        z_gene = torch.cat([z_gc, z_go], dim=-1)

        # 2. Get gene embeddings from Varformer
        z_pvc = self.varformer(
            {
                'pathogenicity': x['pvc']['pathogenicity'],
                'position': x['pvc']['position'],
                'mutation': x['pvc']['mutation']
            },
            mask=mask
        )

        # Apply Gene-Variant Attention
        z_var, variant_attn_weights = self.gene_variant_attention(z_gene, z_pvc)

        # 3. Concatenate processed GC, GO, and PVC features
        concatenated_features = torch.cat([z_gene, z_var], dim=-1)

        # 4. Feed concatenated features into the linear classifier
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x336 and 672x168)
        # TODO: Look at the model architecture in detail
        logits = self.classification_head(concatenated_features).squeeze()
        probabilities = torch.sigmoid(logits)
        binary_predictions = (probabilities > float(self.hyperparams['threshold'])).float()
        if self.hyperparams['return_attn']:
            return logits, probabilities, binary_predictions, variant_attn_weights
        else:
            return logits, probabilities, binary_predictions


class GeneVariantAttention(nn.Module):
    def __init__(self, gene_feature_dim, variant_feature_dim, attention_dim, nhead=1):
        """
        Implements gene-variant attention mechanism.

        Args:
        - gene_feature_dim (int): Number of gene-level features (GC + GO).
        - variant_feature_dim (int): Embedding size of variant features (output of Varformer).
        - attention_dim (int): Attention projection dimension.
        - nhead (int): Number of attention heads (default 1 for interpretability).
        """
        super(GeneVariantAttention, self).__init__()

        self.query_layer = nn.Linear(gene_feature_dim, attention_dim)  # Gene as Query
        self.key_layer = nn.Linear(variant_feature_dim, attention_dim)  # Variant as Key
        self.value_layer = nn.Linear(variant_feature_dim, attention_dim)  # Variant as Value

        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=nhead, batch_first=True)

    def forward(self, gene_features, variant_embeddings):
        """
        Args:
        - gene_features: Tensor [B, gene_feature_dim] – Combined gene context features.
        - variant_embeddings: Tensor [B, S, variant_feature_dim] – Output from Varformer.

        Returns:
        - attn_output: [B, attention_dim] – Variant-informed gene representation.
        - attn_weights: [B, S] – Attention scores over variants.
        """
        B, S, E = variant_embeddings.shape  # B = batch_size, S = max_seq_len (variants per gene)

        # Project features into attention space
        Q = self.query_layer(gene_features).unsqueeze(1)  # Shape: [B, 1, attention_dim]
        K = self.key_layer(variant_embeddings)  # Shape: [B, S, attention_dim]
        V = self.value_layer(variant_embeddings)  # Shape: [B, S, attention_dim]

        # Compute attention (Q attends to K/V)
        attn_output, attn_weights = self.attn(Q, K, V)

        # Remove sequence dimension since each gene has one query
        attn_output = attn_output.squeeze(1)  # [B, attention_dim]
        attn_weights = attn_weights.squeeze(1)  # [B, S]

        return attn_output, attn_weights
