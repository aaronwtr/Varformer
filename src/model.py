import lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef


class PyTorchMLP(torch.nn.Module):
    def __init__(self, config, num_features, num_classes):
        super().__init__()
        self.config = config['mlp']

        self.layers = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(num_features, int(self.config['width_1'])),
            torch.nn.ReLU(),

            # hidden layer
            torch.nn.Linear(int(self.config['width_1']), int(self.config['width_2'])),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(int(self.config['width_2']), 1),
            torch.nn.Sigmoid()
        )

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
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config['mlp']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits, probas, bin_preds = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
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
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))
        return optimizer
