from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = str(Path(__file__).parent.absolute()).split('/')[-1]

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        self.relu = nn.ReLU()

    # compute \mu(x_t), \sigma(x_t), so p(y_t|x_t)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear21(x), self.linear22(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # compute p(x_{t+1}|x_t, z_t)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)

class TD_Cvae(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, z_dim):
        super(TD_Cvae, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim_enc, z_dim)
        self.decoder = Decoder(z_dim+input_dim, hidden_dim_dec, input_dim)
        self.input_dim = input_dim
        self.z_dim = z_dim

    # y_t \sim N(\mu(x_t), \sigma(x_t))
    def _reparameterization_trick(self, mu_x_t, logvar_x_t):
        sigma_t = torch.exp(1/2*logvar_x_t)
        eps = torch.randn_like(sigma_t)
        return mu_x_t + sigma_t*eps

    # z_t = (y_t - y_{t-1}) \sim N(\mu(x_t)-\mu(x_t), \Sigma(x_t)+\Sigma(x_{t-1}))
    def _zrepresentation(self, y_next, y_t):
        return y_next - y_t

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    # compute here D_{KL}[N(\mu(x), \Sigma(x))||N(0,1)]
    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    # inputs: x_t, x_{t+1}
    def forward(self, x_t, x_next):
        mu_x_t, logvar_x_t = self.encoder(x_t)
        mu_x_next, logvar_x_next = self.encoder(x_next)
        y_t = self._reparameterization_trick(mu_x_t, logvar_x_t)
        y_next = self._reparameterization_trick(mu_x_next, logvar_x_next)
        z_t = self._zrepresentation(y_next, y_t)
        # If z_t = 0 then we skip the decoding and just return the input
        xz_t = torch.cat((x_t, z_t), dim=-1)
        x_dec = self.decoder(xz_t) # x_{t+1}
        return x_dec, x_next, mu_x_t-mu_x_next, torch.log(torch.exp(logvar_x_t)+torch.exp(logvar_x_t)), z_t, y_t
        