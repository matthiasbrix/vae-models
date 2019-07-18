import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# - but in merge a pixel may have multiple channels., do stride 1, padding to get same convolution to start with...
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(64*64, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.mean = nn.Linear(1024, 100)
        self.logsigma = nn.Linear(1024, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(-1, np.prod(x.shape))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.mean(x), self.logsigma(x)

# TODO: use maxc poooling here to get it down to correct dims?
# TODO: need padding in decoding, as x_t needs to be adjusted...
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1t = nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1)
        self.conv2t = nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1)
        self.conv3t = nn.ConvTranspose2d(8, 10, 3, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # compute p(x_{t+1}|x_t, z_t)
    def forward(self, x):
        pass

class Tdhcvae(nn.Module):
    def __init__(self, z_dim, beta):
        super(Tdhcvae, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.z_dim = z_dim
        self.beta = beta

    def _reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    # z_t = (y_{t+1} - y_t) \sim N(\mu(x_{t+1})-\mu(x_t), \Sigma(x_{t+1})+\Sigma(x_t))
    def _zrepresentation(self, logvar_x_t, logvar_x_next, mu_x_t, mu_x_next):
        mu_z = mu_x_next - mu_x_t
        logvar_z = torch.log(torch.exp(logvar_x_next)+torch.exp(logvar_x_t))
        z_t = self._reparameterization_trick(mu_z, logvar_z)
        return z_t

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    def loss_function(self, fx, X, logsigma, mu):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + self.beta*kl_divergence, loss_reconstruction, self.beta*kl_divergence

    def forward(self, x_t, x_next):
        mu_x_t, logvar_x_t = self.encoder(x_t)
        mu_x_next, logvar_x_next = self.encoder(x_next)
        z_t = self._zrepresentation(logvar_x_t, logvar_x_next, mu_x_t, mu_x_next)
        print("xt and zt", x_t.shape, z_t.shape)
        x_t = x_t.view(-1, np.prod(x_t.shape))
        print(x_t.shape, z_t.shape)
        xz_t = torch.cat((x_t, z_t), dim=-1)
        x_dec = self.decoder(xz_t)
        exit(1)
        return x_dec, x_next, mu_x_next-mu_x_t, torch.log(torch.exp(logvar_x_next)+torch.exp(logvar_x_t)), z_t, None
        