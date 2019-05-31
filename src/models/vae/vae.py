from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

#########################################################################
#
# Some code is copied from
# https://github.com/pytorch/examples/blob/master/vae/main.py
#
#########################################################################

MODEL_NAME = str(Path(__file__).parent.absolute()).split('/')[-1]

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        self.batch_norm = nn.BatchNorm1d(H)
        self.relu = nn.ReLU()

    # compute q(z|x) which is encoding X into z
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return self.linear21(x), self.linear22(x) # \mu(x), \Sigma(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.batch_norm1 = nn.BatchNorm1d(H)
        self.batch_norm2 = nn.BatchNorm1d(Din)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # compute p(x|z) (posterior) which is decoding to reconstruct x
    def forward(self, z):
        x = self.linear1(z)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.sigmoid(x)

class Vae(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Vae, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    # sampling from N(\mu(x), \Sigma(x))
    def _reparameterization_trick(self, mu, logsigma):
        sigma = torch.exp(1/2*logsigma)
        eps = torch.randn_like(sigma) # sampling eps ~ N(0, I)
        return mu + sigma*eps # compute z = \mu(x) + \Sigma^{1/2}(x) * eps

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    # compute here D_{KL}[N(\mu(x), \Sigma(x))||N(0,1)]
    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    def forward(self, data):
        mu_x, logvar_x = self.encoder(data)
        z = self._reparameterization_trick(mu_x, logvar_x)
        decoded = self.decoder(z)
        return decoded, mu_x, logvar_x, z
