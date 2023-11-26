import torch

import lightning as pl
import torch.nn.functional as F
import xgboost as xgb

from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


class PyTorchMLP(torch.nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        self.config = config['mlp']
        self.layers = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(num_features, int(self.config['width_1'])),
            torch.nn.ReLU(),

            # hidden layer to output layer
            # torch.nn.Linear(int(self.config['width_1']), int(self.config['width_2'])),
            torch.nn.Linear(int(self.config['width_1']), 1),

            # Sigmoid activation is done in loss function
            # torch.nn.Sigmoid()

            # output layer
            # torch.nn.Linear(int(self.config['width_2']), 1),
            # torch.nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()

    def forward(self, x):
        logits = self.layers(x).squeeze()
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits)
        binary_predictions = (probabilities > float(self.config['threshold'])).float()
        return logits, probabilities, binary_predictions


class LightningMLP(pl.LightningModule):
    def __init__(self, model, config, imbalance):
        super().__init__()
        self.model = model
        self.imbalance = imbalance
        self.config = config['mlp']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits, probas, bin_preds = self(features)
        class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                    device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
        self.log('train_loss', loss)
        self.log('train_acc', self.model.train_acc(bin_preds, labels))
        self.log('train_auroc', self.model.train_auroc(bin_preds, labels))
        self.log('train_spearman', self.model.train_spearman(probas, labels.float()))
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits, probas, bin_preds = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log('val_loss', loss)
        self.log('val_acc', self.model.train_acc(bin_preds, labels))
        self.log('val_auroc', self.model.train_auroc(bin_preds, labels))
        self.log('val_spearman', self.model.train_spearman(probas, labels.float()))

    def configure_optimizers(self):
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']))
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']))
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']))
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']))
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")
        lr_start = float(self.config['lr_start'])
        lr_end = float(self.config['lr_end'])
        lr_decay_epochs = int(self.config['epochs'] * 0.5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda epoch: max((lr_end / lr_start) +
                                            (1 - epoch / lr_decay_epochs) * (1 - lr_end / lr_start), lr_end / lr_start))
        return [optimizer], [scheduler]


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
