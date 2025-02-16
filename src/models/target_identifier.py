import torch

import torch.nn as nn

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, Recall, Precision, F1Score
from models.varformer import ShardedVarformer


class BaseTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features, model_type="mlp"):
        super(BaseTargetIdentifier, self).__init__()
        self.config = config['hyperparameters']

        # Create feature extraction layers (everything except the final Linear layer)
        feature_layers = []
        layer_sizes = [num_features] + [int(self.config['width'])] * int(self.config['num_layers'])
        layer_size_prev = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            feature_layers += [
                torch.nn.Linear(layer_size_prev, layer_size),
                torch.nn.BatchNorm1d(layer_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(self.config['dropout']))
            ]
            layer_size_prev = layer_size

        # Store feature extraction layers separately
        self.feature_extractor = torch.nn.Sequential(*feature_layers)

        # Store final classification layer separately
        self.classifier = torch.nn.Linear(layer_sizes[-1], 1)

        # For backwards compatibility, maintain the full sequential model
        self.layers = torch.nn.Sequential(
            self.feature_extractor,
            self.classifier
        )

        self.init_weights = self.initialise_weights()

        self.acc = Accuracy(task="binary", threshold=config['hyperparameters']['threshold'])
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=config['hyperparameters']['threshold'])
        self.precision = Precision(task="binary", threshold=config['hyperparameters']['threshold'])
        self.f1 = F1Score(task="binary", threshold=config['hyperparameters']['threshold'])
        self.spearman = SpearmanCorrCoef()

    def forward(self, x, return_features=False):
        # Extract features
        features = self.feature_extractor(x)

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
    def __init__(self, config, num_features, num_mutations, max_seq_len, num_genes, model_type="varformer"):
        super(VarformerTargetIdentifier, self).__init__(config, num_features, "mlp")

        varformer_config = config['hyperparameters']
        self.varformer = ShardedVarformer(
            max_seq_len=max_seq_len,
            num_muts=num_mutations,
            dropout=float(varformer_config['dropout']),
            d_model=varformer_config['d_model'],
            nhead=varformer_config['nhead'],
            num_encoder_layers=varformer_config['num_encoder_layers']
        )

        self.layers[0] = nn.Linear(varformer_config['d_model'], config['hyperparameters']['width'])

    def forward(self, x, mask=None):
        gene_embeds = self.varformer(x['pathogenicity'], x['position'], x['mutation'], mask)
        logits = self.layers(gene_embeds).squeeze()
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions, gene_embeds


class MultiModalTargetIdentifier(BaseTargetIdentifier):
    def __init__(self, config, num_features_gc, num_features_go, num_mutations, max_seq_len,
                 num_genes):
        super(MultiModalTargetIdentifier, self).__init__(config, num_features_gc + num_features_go + max_seq_len)

        self.num_features_gc = num_features_gc
        self.num_features_go = num_features_go
        self.num_mutations = num_mutations
        self.max_seq_len = max_seq_len
        self.num_genes = num_genes

        self.gc_branch = BaseTargetIdentifier(
            config=config,
            num_features=num_features_gc,
            model_type="mlp"
        )

        self.go_branch = BaseTargetIdentifier(
            config=config,
            num_features=num_features_go,
            model_type="mlp"
        )

        self.pvc_branch = VarformerTargetIdentifier(
            config=config,
            num_features=num_features_gc,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes,
            model_type="varformer"
        )

        # Create the classification branch
        combined_input_size = int(self.config['width']) * 2 + int(self.config['d_model'])
        classification_layers = [
            torch.nn.Linear(combined_input_size, int(self.config['width'])),
            torch.nn.BatchNorm1d(int(self.config['width'])),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=float(self.config['dropout'])),
            torch.nn.Linear(int(self.config['width']), 1)
        ]
        self.classification_branch = torch.nn.Sequential(*classification_layers)

        self.init_weights = self.initialise_weights()

    def forward(self, x, mask=None):
        gc_output_dict = self.gc_branch(x['gc'][0], return_features=True)
        gc_features = gc_output_dict['features']

        go_output_dict = self.go_branch(x['go'][0], return_features=True)
        go_features = go_output_dict['features']

        pvc_logits, pvc_prob, _, pvc_features = self.pvc_branch(
            {
                'pathogenicity': x['pvc']['pathogenicity'],
                'position': x['pvc']['position'],
                'mutation': x['pvc']['mutation']
            },
            mask=mask
        )

        combined_features = torch.cat([gc_features, go_features, pvc_features], dim=-1)

        logits = self.classification_branch(combined_features).squeeze()
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()

        return logits, probabilities, binary_predictions
