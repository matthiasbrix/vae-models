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
        pass

        self.input_dim = Din

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        pass

    # compute p(x|z) (posterior) which is decoding to reconstruct x
    def forward(self, x):
        pass

class Temporal(nn.Module):
    def __init__(self, encoder, decoder):
        super(Temporal, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.states_dict = {} # t -> ([mu(x_t)], [sigma(x_t)])

    # y_t
    def _reparameterization_trick(self, mu, logsigma):
        sigma = torch.exp(1/2*logsigma)
        eps = torch.randn_like(sigma) # sampling eps ~ N(0, I)
        return mu + sigma*eps # compute z = \mu(x) + \Sigma^{1/2}(x) * eps

    # z_t = y_{t+1} - y_t \sim N(\mu(x_{t+1})-\mu(x_t), \Sigma(x_{t+1}+\Sigma(x_t))
    def _zrep(self):
        pass

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    # compute here D_{KL}[N(\mu(x), \Sigma(x))||N(0,1)]
    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    def forward(self, data):
        mu_x, logvar_x = self.encoder(data.view(-1, self.encoder.input_dim))
        z = self._reparameterization_trick(mu_x, logvar_x)
        decoded = self.decoder(z)
        return decoded, mu_x, logvar_x, z
