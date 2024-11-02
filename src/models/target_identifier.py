import torch

import torch.nn as nn

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, Recall
from models.varformer import ShardedVarformer


class BaseTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features, model_type="mlp"):
        super(BaseTargetIdentifier, self).__init__()
        self.config = config['hyperparameters']
        self.layers = []
        layer_sizes = [num_features] + [int(self.config['width'])] * int(self.config['num_layers'])
        layer_size_prev = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            self.layers += [
                torch.nn.Linear(layer_size_prev, layer_size),
                torch.nn.BatchNorm1d(layer_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(self.config['dropout']))
            ]
            layer_size_prev = layer_size
        self.layers += [torch.nn.Linear(layer_sizes[-1], 1)]

        self.layers = torch.nn.Sequential(*self.layers)

        self.init_weights = self.initialise_weights()

        self.acc = Accuracy(task="binary", threshold=config['hyperparameters']['threshold'])
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=config['hyperparameters']['threshold'])
        self.recall_at_10 = Recall(task="binary", threshold=config['hyperparameters']['threshold'], top_k=10)
        self.spearman = SpearmanCorrCoef()

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

    def forward(self, x, mask=None):
        logits = self.layers(x).squeeze()
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


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
    def __init__(self, config, num_features_gc, num_features_go, num_features_pvc, num_mutations, max_seq_len,
                 num_genes):
        super(MultiModalTargetIdentifier, self).__init__(config, num_features_gc + num_features_go + num_features_pvc)

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
            num_features=num_features_pvc,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes,
            model_type="varformer"
        )

        # Create the classification branch
        combined_input_size = int(self.config['width']) * 3
        classification_layers = [
            torch.nn.Linear(combined_input_size, int(self.config['width'])),
            torch.nn.BatchNorm1d(int(self.config['width'])),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=float(self.config['dropout'])),
            torch.nn.Linear(int(self.config['width']), 1)
        ]
        self.classification_branch = torch.nn.Sequential(*classification_layers)

    def forward(self, x, mask=None):
        gc_logits, gc_prob, _ = self.gc_branch(x['gc'])
        gc_features = self.gc_branch.layers[:-1](x['gc'])

        go_logits, go_prob, _ = self.go_branch(x['go'])
        go_features = self.go_branch.layers[:-1](x['go'])

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

    def initialise_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        initial_weights = {}

        gc_weights = self.gc_branch.initialise_weights(seed)
        initial_weights.update({'gc_branch.' + k: v for k, v in gc_weights.items()})

        go_weights = self.go_branch.initialise_weights(seed)
        initial_weights.update({'go_branch.' + k: v for k, v in go_weights.items()})

        pvc_weights = self.pvc_branch.initialise_weights(seed)
        initial_weights.update({'pvc_branch.' + k: v for k, v in pvc_weights.items()})

        for name, module in self.classification_branch.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                initial_weights[f'classification_branch.{name}.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    initial_weights[f'classification_branch.{name}.bias'] = module.bias.detach().clone()
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                initial_weights[f'classification_branch.{name}.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
                    initial_weights[f'classification_branch.{name}.bias'] = module.bias.detach().clone()
                module.running_mean.fill_(0)
                module.running_var.fill_(1)
                initial_weights[f'classification_branch.{name}.running_mean'] = module.running_mean.detach().clone()
                initial_weights[f'classification_branch.{name}.running_var'] = module.running_var.detach().clone()

        return initial_weights
