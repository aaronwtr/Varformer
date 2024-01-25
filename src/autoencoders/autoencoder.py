import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class PathogenicityAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim, num_layers, nhead, reduction_type):
        super(PathogenicityAutoencoder, self).__init__()

        self.reduction_type = reduction_type

        # Define the encoder
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=encoding_dim)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

        # Define the decoder
        decoder_layers = TransformerDecoderLayer(d_model=output_dim, nhead=nhead, dim_feedforward=encoding_dim)
        self.decoder = TransformerDecoder(decoder_layers, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(x, z)
        return y


class AutoencoderTrainer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, encoding_dim, num_layers, nhead, reduction_type):
        super().__init__()
        self.autoencoder = PathogenicityAutoencoder(input_dim, output_dim, encoding_dim,num_layers, nhead,
                                                    reduction_type)
        self.reduction_type = reduction_type
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.autoencoder(x)

    def pathogenicity_embedding(self, x):
        z = self.autoencoder.encoder(x)
        return z

    def predict_step(self, batch):
        x = batch
        z = self.autoencoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss_fn(x_hat, x)
        log_loss = math.log(loss, 10)
        self.log('reconstruction_loss', loss)
        self.log('log_reconstruction_loss', log_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
