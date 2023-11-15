import lightning as pl
import torch
import torch.nn.functional as F


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
            torch.nn.Linear(int(self.config['width_1']), num_classes)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class LightningMLP(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config['mlp']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))
        return optimizer
