import torch
import pytorch_lightning as pl
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim * 2)  # Output mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.input_dim),
            nn.Sigmoid()  # To keep output values between 0 and 1
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, loss_type='bce'):
        if loss_type == 'bce':
            recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss


class VAETrainer(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim)

    def forward(self, x):
        # Encoding
        mu, logvar = torch.chunk(self.vae.encoder(x), 2, dim=1)

        # Reparameterization trick (sample from the distribution)
        z = self.vae.reparameterize(mu, logvar)

        # Decoding
        reconstruction = self.vae.decoder(z)
        return reconstruction, mu, logvar, z

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar, _ = self(x)
        loss = self.vae.loss_function(x_hat, x, mu, logvar)
        self.log('reconstruction_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        x_hat, mu, logvar, z = self(x)
        return z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
