import torch.utils.data
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, kernel_size_high, kernel_size_low, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, z_dim, kernel_size_high, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, kernel_size_high, stride=1, padding=1)
        self.conv3 = nn.Conv2d(z_dim, z_dim, kernel_size_low, stride=1)
        self.mean = nn.Conv2d(z_dim, z_dim, kernel_size_low, stride=1)
        self.logsigma = nn.Conv2d(z_dim, z_dim, kernel_size_low, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Computes mu(x_t), sigma(x_t), so q(y_t|x_t)."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.relu(self.mean(x)), self.relu(self.logsigma(x))

class Decoder(nn.Module):
    def __init__(self, kernel_size_high, kernel_size_low, z_dim):
        super(Decoder, self).__init__()
        self.conv1t = nn.Conv2d(1+z_dim, z_dim, kernel_size_low, stride=1)
        self.conv2t = nn.Conv2d(z_dim, z_dim, kernel_size_low, stride=1)
        self.conv3t = nn.Conv2d(z_dim, z_dim, kernel_size_high, stride=1, padding=1)
        self.conv4t = nn.Conv2d(z_dim, 1, kernel_size_high, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Computes p(x_{t+1}|x_t, z_t)"""
        x = self.relu(self.conv1t(x))
        x = self.relu(self.conv2t(x))
        x = self.relu(self.conv3t(x))
        x = self.conv4t(x)
        return self.sigmoid(x)

class Tdcvae2(nn.Module):
    def __init__(self, z_dim, beta, kernel_size_high, kernel_size_low):
        super(Tdcvae2, self).__init__()
        self.encoder = Encoder(kernel_size_high, kernel_size_low, z_dim)
        self.decoder = Decoder(kernel_size_high, kernel_size_low, z_dim)
        self.z_dim = z_dim
        self.beta = beta
        self.loss = nn.BCELoss(reduction="sum")

    def _reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    def _zrepresentation(self, logvar_x_t, logvar_x_next, mu_x_t, mu_x_next):
        mu_z = mu_x_next - mu_x_t
        logvar_z = torch.log(torch.exp(logvar_x_next)+torch.exp(logvar_x_t))
        z_t = self._reparameterization_trick(mu_z, logvar_z)
        return z_t, mu_z, logvar_z

    def loss_function(self, fx, X, logsigma, mu):
        """Represents the objective function of the model (reconstruction loss + KL divergence).

        Args:
            fx: The reconstruction.
            X: The ground truth input.
            logsigma: gamma(x) (so log variances)
            mu: mu(x) of the model.
        Returns:
            The objective outputs, reconstruction loss, kl divergence
        """
        loss_reconstruction = self.loss(fx, X)
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1. - logsigma)
        return loss_reconstruction + self.beta*kl_divergence, loss_reconstruction, kl_divergence

    def forward(self, x_t, x_next):
        """Computes the encodings of x_t, x_{t+1} and decoding of x_{t+1}.

        Args:
            x_t: the image x_t
            x_next: the image x_{t+1} (default: None)

        Note:
            Shapes of x_t and x_{t+1} should both be N x C x H x W, otherwise exit.
        """
        if len(x_t.shape) is not 4 and len(x_next.shape) is not 4:
            raise ValueError("Need shape of x_t and x_{t+1} to be N x C x H x W ")
        mu_x_t, logvar_x_t = self.encoder(x_t)
        mu_x_next, logvar_x_next = self.encoder(x_next)
        z_t, mu_z, logvar_z = self._zrepresentation(logvar_x_t, logvar_x_next, mu_x_t, mu_x_next)
        xz_t = torch.cat((x_t, z_t), dim=1)
        x_dec = self.decoder(xz_t)
        return x_dec, x_next, mu_z, logvar_z, z_t, mu_x_t

