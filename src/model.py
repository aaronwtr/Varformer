import torch

import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, Precision
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


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

        self.acc = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
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

    def forward(self, x):
        logits = self.layers(x).squeeze()
        # print("\nInput:", x)
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     print(f"Output of layer {i}:", x)
        #
        # logits = x.squeeze()
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, imbalance, model_type="mlp"):
        super().__init__()
        self.imbalance = imbalance
        self.config = config['hyperparameters'][model_type]
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits, probas, bin_preds = self(features)
        class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                    device=self.device)
        labels = (labels > float(self.config['threshold'])).float()

        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)

        self.log('train_loss', loss)
        self.log('train_acc', self.model.acc(bin_preds, labels))
        self.log('train_auroc', self.model.auroc(bin_preds, labels))
        self.log('train_spearman', self.model.spearman(probas, labels.float()))
        self.log('train_precision', self.model.precision(bin_preds, labels))

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits, probas, bin_preds = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        labels = (labels > float(self.config['threshold'])).float()

        self.log('val_loss', loss)
        self.log('val_acc', self.model.acc(bin_preds, labels))
        self.log('val_auroc', self.model.auroc(bin_preds, labels))
        self.log('val_spearman', self.model.spearman(probas, labels.float()))
        self.log('val_precision', self.model.precision(bin_preds, labels))

    def test_step(self, batch, batch_idx):
        features, labels, test_source = batch
        logits, probas, bin_preds = self(features)
        labels = (labels > float(self.config['threshold'])).float()
        test_source = test_source[0]

        self.log(f'test_acc_{test_source}', self.model.acc(bin_preds, labels))
        self.log(f'test_auroc_{test_source}', self.model.auroc(bin_preds, labels))
        self.log(f'test_spearman_{test_source}', self.model.spearman(probas, labels.float()))
        self.log(f'test_precision_{test_source}', self.model.precision(bin_preds, labels))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        logits, probas, bin_preds = self(features)
        return probas

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
