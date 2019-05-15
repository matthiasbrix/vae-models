from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

PATH = str(Path(__file__).parent.absolute()).split('/')[-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        self.batch_norm = nn.BatchNorm1d(H)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return self.linear21(x), self.linear22(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.batch_norm1 = nn.BatchNorm1d(H)
        self.batch_norm2 = nn.BatchNorm1d(Din)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.linear1(z)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.sigmoid(x)

class Cvae(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, y_size):
        super(Cvae, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim+y_size, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim+y_size, hidden_dim, input_dim)
        self.y_size = y_size

    def onehot_encoding(self, y):
        y = y.view(y.size(0), 1).type(torch.LongTensor)
        onehot = torch.zeros(y.size(0), self.y_size, dtype=torch.float, device=device) # batch_size x y_size
        onehot.scatter_(1, y, 1)
        return onehot

    def reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    def forward(self, x, y=None):
        y_one_hot = self.onehot_encoding(y) # batch_size x y_size
        x = x.view(-1, self.input_dim) # from batch_size x 1 x x.x x x.y to batch_size x x.x*x.y
        x = torch.cat((x, y_one_hot), dim=-1) # batch_size x (x.y+y_one_hot.y)
        mu_x, logvar_x = self.encoder(x)
        latent_space = self.reparameterization_trick(mu_x, logvar_x)
        z = torch.cat((latent_space, y_one_hot), dim=-1)
        decoded = self.decoder(z)
        return decoded, mu_x, logvar_x, latent_space
