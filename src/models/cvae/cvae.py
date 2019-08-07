from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = str(Path(__file__).parent.absolute()).split('/')[-1]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout, batch_norm_flag):
        super(Encoder, self).__init__()
        self.batch_norm_flag = batch_norm_flag
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        if self.batch_norm_flag:
            self.batch_norm = nn.BatchNorm1d(H)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x) if self.batch_norm_flag else x
        x = self.relu(x)
        return self.linear21(x), self.linear22(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din, batch_norm_flag):
        super(Decoder, self).__init__()
        self.batch_norm_flag = batch_norm_flag
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        if self.batch_norm_flag:
            self.batch_norm1 = nn.BatchNorm1d(H)
            self.batch_norm2 = nn.BatchNorm1d(Din)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.linear1(z)
        x = self.batch_norm1(x) if self.batch_norm_flag else x
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x) if self.batch_norm_flag else x
        return self.sigmoid(x)

class Cvae(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, beta, y_size, batch_norm_flag):
        super(Cvae, self).__init__()
        self.encoder = Encoder(input_dim+y_size, hidden_dim, z_dim, batch_norm_flag)
        self.decoder = Decoder(z_dim+y_size, hidden_dim, input_dim, batch_norm_flag)
        self.y_size = y_size
        self.z_dim = z_dim
        self.beta = beta

    def onehot_encoding(self, y):
        y = y.view(y.size(0), 1).type(torch.LongTensor).to(DEVICE)
        onehot = torch.zeros(y.size(0), self.y_size, dtype=torch.float, device=DEVICE) # batch_size x y_size
        onehot.scatter_(1, y, 1)
        return onehot

    def reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    def loss_function(self, fx, X, logsigma, mu):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + self.beta*kl_divergence, loss_reconstruction, self.beta*kl_divergence

    def forward(self, x, y=None):
        y_one_hot = self.onehot_encoding(y) # batch_size x y_size
        x = torch.cat((x, y_one_hot), dim=-1) # batch_size x (x.y+y_one_hot.y)
        mu_x, logvar_x = self.encoder(x)
        latent_space = self.reparameterization_trick(mu_x, logvar_x)
        z = torch.cat((latent_space, y_one_hot), dim=-1)
        decoded = self.decoder(z)
        return decoded, mu_x, logvar_x, latent_space
