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

    def reduction(self, x):
        if self.reduction_type == "padding":
            x = self.padding(x)
        elif self.reduction_type == "pooling":
            x = self.pooling(x)
        else:
            raise ValueError("Invalid reduction type. Expected 'padding' or 'pooling'.")
        return x

    def padding(self, x):
        current_dimension = x.size(-1)
        if current_dimension == self.output_dim:
            return x
        elif current_dimension < self.output_dim:
            # Calculate the required padding on both sides
            padding_left = (self.output_dim - current_dimension) // 2
            padding_right = self.output_dim - current_dimension - padding_left

            # Pad the tensor
            padded_x = F.pad(x, (padding_left, padding_right), value=0)
            return padded_x
        else:
            raise ValueError("Current dimension is already greater than the target dimension.")

    def pooling(self, x):
        current_dimension = x.size(-1)

        if current_dimension == self.output_dim:
            return x
        elif current_dimension > self.output_dim:
            pool = nn.AdaptiveAvgPool1d(self.output_dim)
            pooled_x = pool(x)
            return pooled_x
        else:
            raise ValueError("Current dimension is already smaller than the target dimension.")
