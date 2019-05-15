from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

PATH = str(Path(__file__).parent.absolute()).split('/')[-1]

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        pass

    # compute \mu(x_t), \sigma(x_t), so p(y_t|x_t)
    def forward(self, x_t):
        pass

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        pass

    # compute p(x_{t+1}|x_t, z_{t+1})
    def forward(self, x_t):
        pass

class TDVae(nn.Module):
    def __init__(self, data_loader.input_dim, hidden_dim, z_dim, T):
        super(Temporal, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)
        self.states_dict = {} # t -> (mu(x_t), sigma(x_t), y_t)
        self.T = T

    # y_t \sim N(\mu(x_t), \sigma(x_t))
    def _reparameterization_trick(self, mu_x_t, logvar_x_t):
        sigma_t = torch.exp(1/2*logvar_x_t)
        eps = torch.randn_like(sigma)
        return mu_x_t + sigma_t*eps

    # z_t = (y_t - y_{t-1}) \sim N(\mu(x_t)-\mu(x_t), \Sigma(x_t-\Sigma(x_{t-1}))
    def _zrepr(self, t):
        _, _, y_t = self.states_dict[t]
        _, _, y_prev = self.states_dict[t-1]
        return y_t - y_prev

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    # compute here D_{KL}[N(\mu(x), \Sigma(x))||N(0,1)]
    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    # at start t=0, so x_0 is input
    def forward(self, x_t):
        mu_x_t, logvar_x_t = self.encoder(x_t.view(-1, self.encoder.input_dim))
        y_t = self._reparameterization_trick(mu_x_t, logvar_x_t)
        self.states_dict[t] = (mu_x_t, logvar_x_t, y_t)
        z_t = self._zrepr(t) if self.t >= 1 else None
        # If z_t = 0 then we skip the decoding and just return the input
        if not z_t or not np.any(z_t.cpu().detach().numpy()):
            x_t = x_t
        else:
            # TODO: concat x_t and z_t
            x_t = self.decoder(x_t) # x_{t+1}
        return decoded, mu_x, logvar_x, z_t
