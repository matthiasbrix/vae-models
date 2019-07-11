import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# TODO: input data spatial dimensions? AND CHANNELS?
# TODO: 64x64
class Encoder(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size, stride=4, padding=3)
        self.conv2 = nn.Conv2d(8, 3, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(20, 20)
        self.linear2 = nn.Linear(20, 20)

    # compute \mu(x_t), \sigma(x_t), so q(y_t|x_t)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return self.linear1(x), self.linear2(x)

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
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, z_dim, beta):
        super(Tdhcvae, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim_enc, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim_dec, input_dim)
        self.z_dim = z_dim
        self.beta = beta

    # y \sim N(\mu(x), \Sigma(x))
    def _reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    # z_t = (y_{t+1} - y_t) \sim N(\mu(x_{t+1})-\mu(x_t), \Sigma(x_{t+1})+\Sigma(x_t))
    def _zrepresentation(self, y_next, y_t):
        return y_next - y_t

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    def loss_function(self, fx, X, logsigma, mu):
        pass

    # inputs: x_t, x_{t+1}
    def forward(self, x_t, x_next=None):
        pass
        