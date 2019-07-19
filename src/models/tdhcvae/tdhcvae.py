import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: take into account the batch size.... AND CROPPED 
# TODO: AND CRoPPED SIZE1!!
IN1 = 1
IN2 = 3
IN3 = 5
IN4 = 7

# - but in merge a pixel may have multiple channels., do stride 1, padding to get same convolution to start with...
class Encoder(nn.Module):
    def __init__(self, input_dim, kernel_size, z_dim):
        super(Encoder, self).__init__()
        num_pixels = np.prod(input_dim)
        self.conv1 = nn.Conv2d(IN1, IN2, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(IN2, IN3, kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(IN3, IN4, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(IN4*num_pixels, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.mean = nn.Linear(1024, z_dim)
        self.logsigma = nn.Linear(1024, z_dim)

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
    def __init__(self, input_dim, kernel_size, z_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        num_pixels = np.prod(input_dim)
        self.linear1 = nn.Linear(num_pixels + z_dim, num_pixels + z_dim)
        self.linear2 = nn.Linear(num_pixels + z_dim, IN4*num_pixels)
        self.conv1t = nn.ConvTranspose2d(IN4, IN3, kernel_size, stride=1, padding=1)
        self.conv2t = nn.ConvTranspose2d(IN3, IN2, kernel_size, stride=1, padding=1)
        self.conv3t = nn.ConvTranspose2d(IN2, IN1, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # compute p(x_{t+1}|x_t, z_t)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        #print(x.shape)
        x = x.view((1, IN4, *self.input_dim[1:]))
        #print(x.shape)
        x = self.conv1t(x)
        x = self.relu(x)
        x = self.conv2t(x)
        x = self.relu(x)
        x = self.conv3t(x)
        x = self.sigmoid(x)
        return x

class Tdhcvae(nn.Module):
    def __init__(self, z_dim, beta, kernel_size, input_dim):
        super(Tdhcvae, self).__init__()
        print(input_dim)
        self.encoder = Encoder(input_dim, kernel_size, z_dim)
        self.decoder = Decoder(input_dim, kernel_size, z_dim)
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
        #print("start model!", x_t.shape)
        mu_x_t, logvar_x_t = self.encoder(x_t)
        mu_x_next, logvar_x_next = self.encoder(x_next) # TODO: is same convolution right now...
        z_t = self._zrepresentation(logvar_x_t, logvar_x_next, mu_x_t, mu_x_next)
        #print("xt and zt", x_t.shape, z_t.shape)
        x_t = x_t.view(-1, np.prod(x_t.shape))
        #print(x_t.shape, z_t.shape)
        xz_t = torch.cat((x_t, z_t), dim=-1)
        #print(xz_t.shape)
        x_dec = self.decoder(xz_t)
        #print(x_dec.shape)
        #exit(1)
        return x_dec, x_next, mu_x_next-mu_x_t, torch.log(torch.exp(logvar_x_next)+torch.exp(logvar_x_t)), z_t, None
        