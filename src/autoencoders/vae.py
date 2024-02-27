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
            nn.Linear(self.input_dim, self.latent_dim),  # Assuming MNIST images (28*28=784)
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

    def forward(self, x):
        # Encoding
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)

        # Reparameterization trick (sample from the distribution)
        z = self.reparameterize(mu, logvar)

        # Decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        # Use MSE loss for reconstruction
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss


class VAETrainer(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        return x_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
