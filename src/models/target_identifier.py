import torch

import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, F1Score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from varformer import Varformer, ShardedVarformer, VariantAttention


class BaseTargetIdentifier(torch.nn.Module):
    def __init__(self, config, num_features, model_type="mlp"):
        super(BaseTargetIdentifier, self).__init__()
        self.config = config['hyperparameters'][model_type]
        self.layers = []
        layer_sizes = [num_features] + [int(self.config['width'])] * int(self.config['depth'])
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
        varformer_output = self.varformer(pat, pos, mut, mask)
        logits = self.layers(varformer_output).squeeze()
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


class ShardedVarformerTargetIdentifier(BaseTargetIdentifier):
    def __init__(self, config, num_features, num_mutations, max_seq_len, model_type="varformer"):
        super(ShardedVarformerTargetIdentifier, self).__init__(config, num_features, "mlp")

        varformer_config = config['hyperparameters'][model_type]
        self.varformer = ShardedVarformer(
            max_seq_len=max_seq_len,
            num_muts=num_mutations,
            d_model=varformer_config['d_model'],
            nhead=varformer_config['nhead'],
            num_encoder_layers=varformer_config['num_layers']
        )
        self.aggregator = GeneAggregator(varformer_config['d_model'], varformer_config['nhead'])

        # Adjust the input size of the first linear layer of the BaseTargetIdentifier MLP
        self.layers[0] = nn.Linear(varformer_config['d_model'], int(self.config['width']))

    def forward(self, x, mask=None):
        shard_embeds = self.varformer(
            x['pathogenicity'],
            x['position'],
            x['mutation'],
            mask
        )
        gene_embed = self.aggregator(shard_embeds)
        logits = self.layers(gene_embed).squeeze()
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, imbalance, model_type):
        super().__init__()
        self.imbalance = imbalance
        self.config = config['hyperparameters'][model_type]
        self.model = model

    def _common_step(self, batch, batch_idx, step_type):
        if len(batch) > 2:  # For transformer
            features = {key: batch[key] for key in ['pathogenicity', 'position', 'mutation']}
            labels = batch['labels']
            masks = batch['mask']
            gene_ids = batch['gene_id']
            shard_id = batch['shard_id']
            total_shards = batch['total_shards']

            logits, probas, bin_preds = self(features, masks)
        else:  # For regular MLP
            features, labels = batch
            logits, probas, bin_preds = self(features)

        if step_type == 'train':
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        labels = (labels > float(self.config['threshold'])).float()

        self.log(f'{step_type}_loss', loss)
        self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels))
        self.log(f'{step_type}_auroc', self.model.auroc(bin_preds, labels.int()))
        self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()))
        self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()))

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:  # For varformer
            features, masks, labels, test_source = batch
            logits, probas, bin_preds = self(features, masks)
        else:  # For regular MLP
            features, labels, test_source = batch
            logits, probas, bin_preds = self(features)

        labels = (labels > float(self.config['threshold'])).float()
        test_source = test_source[0]

        self.log(f'test_acc_{test_source}', self.model.acc(bin_preds, labels))
        self.log(f'test_auroc_{test_source}', self.model.auroc(bin_preds, labels.int()))
        self.log(f'test_spearman_{test_source}', self.model.spearman(probas, labels.float()))
        self.log(f'test_f1_{test_source}', self.model.f1(bin_preds, labels.long()))

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        return optimizer


class MLPLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance, model_type="mlp"):
        super().__init__(model, config, imbalance, model_type)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        logits, probas, bin_preds = self(features)
        return probas


class VarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance, model_type="varformer"):
        super().__init__(model, config, imbalance, model_type)

    def forward(self, x, mask):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, masks = batch
        logits, probas, bin_preds = self(features, masks)
        return probas


class ShardedVarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance, model_type="varformer"):
        super().__init__(model, config, imbalance, model_type)
        self.model = model
        self.accumulated_shards = {}

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        shard_embeds = self(batch)

        # Accumulate shard embeddings
        for i, (gene_id, shard_id) in enumerate(zip(batch['gene_id'], batch['shard_id'])):
            if gene_id not in self.accumulated_shards:
                self.accumulated_shards[gene_id] = {}
            self.accumulated_shards[gene_id][shard_id] = shard_embeds[i]

        # Process genes with all shards present
        complete_genes = []
        for gene_id, shards in self.accumulated_shards.items():
            if len(shards) == batch['total_shards'][batch['gene_id'] == gene_id][0]:
                gene_embed = self.model.aggregator(torch.stack(list(shards.values())))
                complete_genes.append((gene_id, gene_embed))
                del self.accumulated_shards[gene_id]

        if complete_genes:
            gene_ids, gene_embeds = zip(*complete_genes)
            gene_embeds = torch.stack(gene_embeds)

            # Compute loss based on your specific task
            # TODO: make sure the below is the targetidentifier
            logits, probas, bin_preds = self.model.layers(gene_embeds).squeeze()
            class_weight = torch.tensor([1 if batch['labels'][i] == 0 else self.imbalance for i in range(len(batch['labels']))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, batch['labels'].float(), weight=class_weight)
            self.log('train_loss', loss)
            return loss
        else:
            return None  # Skip the optimizer step when no complete genes are available

    def on_train_epoch_end(self):
        # Clear accumulated shards at the end of each epoch
        self.accumulated_shards.clear()

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        return optimizer


class GeneAggregator(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.attention = VariantAttention(d_model, nhead)

    def forward(self, shard_embeds):
        # shard_embeds shape: (num_shards, d_model)
        shard_embeds = shard_embeds.unsqueeze(1)  # (num_shards, 1, d_model)
        output = self.attention(shard_embeds, shard_embeds, shard_embeds)
        return output.mean(0)


class VariantRepresentationTargetIdentifier(pl.LightningModule):
    def __init__(self, vae, vae_epochs, base_model, config, num_features, latent_dim, imbalance):
        super().__init__()
        self.vae = vae
        self.vae_epochs = vae_epochs
        self.base_model = base_model
        self.config = config
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.imbalance = imbalance

    def forward(self, x):
        x_arr = x.detach().numpy()
        mu, logvar = torch.chunk(self.vae.encoder(x), 2, dim=1)
        z = self.vae.reparameterize(mu, logvar)
        z_arr = z.detach().numpy()
        reconstruction = self.vae.decoder(z)
        logits, probabilities, binary_predictions = self.base_model(z)
        return reconstruction, mu, logvar, logits, probabilities, binary_predictions

    def training_step(self, batch, batch_idx):
        features, labels = batch
        reconstruction, mu, logvar, logits, probas, bin_preds = self(features)
        if self.current_epoch < self.vae_epochs:
            vae_loss = self.vae.loss_function(reconstruction, features, mu, logvar)
            self.log('vae_loss', vae_loss)
            return vae_loss
        else:
            vae_loss = self.vae.loss_function(reconstruction, features, mu, logvar)
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                    device=self.device)
            labels = (labels > float(self.config['hyperparameters']['mlp']['threshold'])).float()
            prediction_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
            loss = vae_loss + prediction_loss
            self.log('loss', loss)
            self.log('vae_loss', vae_loss)
            self.log('prediction_loss', prediction_loss)
            self.log('train_acc', self.base_model.acc(bin_preds, labels))
            self.log('train_auroc', self.base_model.auroc(bin_preds, labels.int()))
            self.log('train_spearman', self.base_model.spearman(probas, labels.float()))
            self.log('train_f1', self.base_model.f1(bin_preds, labels.long()))
            return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        reconstruction, mu, logvar, logits, probas, bin_preds = self(features)
        if self.current_epoch < self.vae_epochs:
            vae_loss = self.vae.loss_function(reconstruction, features, mu, logvar)
            self.log('vae_loss', vae_loss)
            return vae_loss
        else:
            vae_loss = self.vae.loss_function(reconstruction, features, mu, logvar)
            prediction_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            labels = (labels > float(self.config['hyperparameters']['mlp']['threshold'])).float()
            loss = vae_loss + prediction_loss
            self.log('val_loss', loss)
            self.log('val_acc', self.base_model.acc(bin_preds, labels))
            self.log('val_auroc', self.base_model.auroc(bin_preds, labels.int()))
            self.log('val_spearman', self.base_model.spearman(probas, labels.float()))
            self.log('val_f1', self.base_model.f1(bin_preds, labels.long()))
            return loss

    def test_step(self, batch, batch_idx):
        features, labels, test_source = batch
        reconstruction, mu, logvar, logits, probas, bin_preds = self(features)
        labels = (labels > float(self.config['hyperparameters']['mlp']['threshold'])).float()
        test_source = test_source[0]
        self.log(f'test_acc_{test_source}', self.base_model.acc(bin_preds, labels))
        self.log(f'test_auroc_{test_source}', self.base_model.auroc(bin_preds, labels.int()))
        self.log(f'test_spearman_{test_source}', self.base_model.spearman(probas, labels.float()))
        self.log(f'test_f1_{test_source}', self.base_model.f1(bin_preds, labels.long()))

    def configure_optimizers(self):
        weight_decay = float(self.config['hyperparameters']['mlp'].get('weight_decay', 0))
        if self.config['hyperparameters']['mlp']['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['hyperparameters']['mlp']['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['hyperparameters']['mlp']['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['hyperparameters']['mlp']['lr_start']),
                                        weight_decay=weight_decay)
        elif self.config['hyperparameters']['mlp']['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['hyperparameters']['mlp']['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['hyperparameters']['mlp']['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['hyperparameters']['mlp']['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['hyperparameters']['mlp']['optimizer']} not recognized.")
        return optimizer


class EnsembleTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, models, config, imbalance, model_type="mlp"):
        super().__init__(config, imbalance, model_type)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        combined_predictions = torch.mean(torch.stack(predictions), dim=0)
        return combined_predictions


class XGBoostModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': 0.1,
                'max_depth': 5,
                'n_estimators': 100
            }
        else:
            self.params = params
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        eval_set = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(self.params, dtrain, num_boost_round=1000, evals=eval_set, early_stopping_rounds=10)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred > 0.5), roc_auc_score(y, y_pred), spearmanr(y, y_pred)[0]
