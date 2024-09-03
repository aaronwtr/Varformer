import torch

import torch.nn as nn

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, F1Score
from models.varformer import Varformer, ShardedVarformer, GeneAggregator


class BaseTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features, model_type="mlp"):
        super(BaseTargetIdentifier, self).__init__()
        self.config = config['hyperparameters'][model_type]
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

        self.acc = Accuracy(task="binary", threshold=config['hyperparameters'][model_type]['threshold'])
        self.auroc = AUROC(task="binary")
        self.f1 = F1Score(task="binary", threshold=config['hyperparameters'][model_type]['threshold'])
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
    def __init__(self, config, num_features, num_mutations, max_seq_len, model_type="varformer"):
        super(VarformerTargetIdentifier, self).__init__(config, num_features, "mlp")

        varformer_config = config['hyperparameters'][model_type]
        self.varformer = Varformer(
            num_mutations=num_mutations,
            max_seq_length=max_seq_len,
            d_model=varformer_config['d_model'],
            nhead=varformer_config['nhead'],
            num_layers=varformer_config['num_layers']
        )

        # Adjust the input size of the first linear layer of the BaseTargetIdentifier MLP
        self.layers[0] = nn.Linear(varformer_config['d_model'], int(self.config['width']))

    def forward(self, x, mask=None):
        pat = x['pathogenicity']
        pos = x['position']
        mut = x['mutation']
        gene = x['gene']
        varformer_output = self.varformer(pat, pos, mut, gene, mask)
        logits = self.layers(varformer_output).squeeze()
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


class ShardedVarformerTargetIdentifier(BaseTargetIdentifier):
    def __init__(self, config, num_features, num_mutations, max_seq_len, num_genes, model_type="varformer"):
        super(ShardedVarformerTargetIdentifier, self).__init__(config, num_features, "mlp")

        varformer_config = config['hyperparameters'][model_type]
        self.varformer = ShardedVarformer(
            max_seq_len=max_seq_len,
            num_muts=num_mutations,
            num_genes=num_genes,
            dropout=float(varformer_config['dropout']),
            shard_size=varformer_config['shard_size'],
            nhead=varformer_config['nhead'],
            num_encoder_layers=varformer_config['num_encoder_layers']
        )
        # self.aggregator = GeneAggregator(varformer_config['d_model'], varformer_config['nhead'])

        # d_model = varformer_config['shard_size'] // 3 + varformer_config['shard_size']
        # Adjust the input size of the first linear layer of the BaseTargetIdentifier MLP
        self.layers[0] = nn.Linear(varformer_config['shard_size'], config['hyperparameters']['mlp']['width'])

    def forward(self, x, mask=None):
        shard_embeds = self.varformer(x['pathogenicity'], x['position'], x['mutation'], x['gene'], mask)
        # gene_embeds = self.aggregator(shard_embeds).squeeze(1)
        logits = self.layers(shard_embeds).squeeze()
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions, shard_embeds
