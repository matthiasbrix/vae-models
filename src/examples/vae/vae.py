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

PATH = str(Path(__file__).parent.absolute()).split('/')[-1]

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        self.bn = nn.BatchNorm1d(H)
        self.input_dim = Din

    # compute q(z|x) which is encoding X into z
    def forward(self, x):
        x = self.linear1(x)
        #x = self.bn(x)
        x = F.relu(x)
        return self.linear21(x), self.linear22(x) # \mu(x), \Sigma(x) so mean(x) and covariance(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.bn = nn.BatchNorm1d(H)
        self.bn2 = nn.BatchNorm1d(Din)
        self.sigmoid = nn.Sigmoid()

    # compute p(x|z) (posterior) which is decoding to reconstruct X
    def forward(self, x):
        x = self.linear1(x)
        #x = self.bn(x)
        x = F.relu(x)
        x = self.linear2(x)
        #x = self.bn2(x)
        return self.sigmoid(x)

class Vae(nn.Module):
    def __init__(self, encoder, decoder):
        super(Vae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # sampling from N(\mu(X), \Sigma(X))
    def reparameterization_trick(self, mu, logsigma):
        sigma = torch.exp(1/2*logsigma)
        eps = torch.randn_like(sigma) # sampling eps ~ N(0, I)
        return mu + sigma*eps # compute z = \mu(X) + \Sigma^{1/2}(X) * eps

    # loss function + KL divergence, use for this \mu(X), \Sigma(X)
    # compute here D_{KL}[N(\mu(X), \Sigma(X))||N(0,1)] = 1/2 \sum_k (\Sigma(X)+\mu^2(X) - 1 - log \Sigma(X))
    def loss_function(self, fx, X, logsigma, mu):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) # by appendix B in the Auto Encoding Variational Bayes
        #kl_divergence2 = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma) # will give same value but negative would need to + below
        return loss_reconstruction - kl_divergence, loss_reconstruction, -kl_divergence

    def forward(self, data):
        mu, logsigma = self.encoder(data.view(-1, self.encoder.input_dim))
        z = self.reparameterization_trick(mu, logsigma)
        decoded = self.decoder(z)
        return decoded, mu, logsigma, z
