import torch
import torch.utils.data
import torch.nn as nn

class EncoderRotation(nn.Module):
    def __init__(self, Din, Dout):
        super(EncoderRotation, self).__init__()
        self.linear1 = nn.Linear(Din, 500)
        self.linear2 = nn.Linear(500, 200)
        self.mean = nn.Linear(200, Dout)
        self.logsigma = nn.Linear(200, Dout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.mean(x), self.logsigma(x)

class EncoderScaling(nn.Module):
    def __init__(self, Din, Dout):
        super(EncoderScaling, self).__init__()
        self.linear1 = nn.Linear(Din, 500)
        self.linear2 = nn.Linear(500, 200)
        self.mean = nn.Linear(200, Dout)
        self.logsigma = nn.Linear(200, Dout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.mean(x), self.logsigma(x)

class EncoderRotationScaling(nn.Module):
    def __init__(self, Din, Dout):
        super(EncoderRotationScaling, self).__init__()
        self.linear1 = nn.Linear(Din, 98)
        self.linear2 = nn.Linear(98, 12)
        self.linear3 = nn.Linear(12, 12)
        self.mean = nn.Linear(12, Dout)
        self.logsigma = nn.Linear(12, Dout)
        self.relu = nn.ReLU()

    # compute \mu(x_t), \sigma(x_t), so q(y_t|x_t)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.mean(x), self.logsigma(x)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din, rotations):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, Din)
        self.relu = nn.ReLU()
        self.layers = 7 if rotations else 3
        self.sigmoid = nn.Sigmoid()

    # compute p(x_{t+1}|x_t, z_t)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        for _ in range(self.layers):
            x = self.relu(self.linear2(x))
        return self.sigmoid(self.linear3(x))

class TD_Cvae(nn.Module):
    def __init__(self, input_dim, hidden_dim_dec, z_dim, beta, rotations=False, scaling=False):
        super(TD_Cvae, self).__init__()
        if rotations and scaling:
            self.encoder = EncoderRotationScaling(input_dim, z_dim)
        elif rotations:
            self.encoder = EncoderRotation(input_dim, z_dim)
        elif scaling:
            self.encoder = EncoderScaling(input_dim, z_dim)
        self.decoder = Decoder(input_dim+z_dim, hidden_dim_dec+z_dim, input_dim, rotations)
        self.z_dim = z_dim
        self.beta = beta
        self.loss = nn.MSELoss() if rotations and scaling else nn.BCELoss(reduction="sum")

    # y \sim N(\mu(x), \Sigma(x))   
    def _reparameterization_trick(self, mu_x, logvar_x):
        sigma = torch.exp(1/2*logvar_x)
        eps = torch.randn_like(sigma)
        return mu_x + sigma*eps

    # z_t = (y_{t+1} - y_t) \sim N(\mu(x_{t+1})-\mu(x_t), \Sigma(x_{t+1})+\Sigma(x_t))
    def _zrepresentation(self, logvar_x_t, logvar_x_next, mu_x_t, mu_x_next):
        mu_z = mu_x_next - mu_x_t
        logvar_z = torch.log(torch.exp(logvar_x_next)+torch.exp(logvar_x_t))
        z_t = self._reparameterization_trick(mu_z, logvar_z)
        return z_t, mu_z, logvar_z

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    def loss_function(self, fx, X, logsigma, mu):
        loss_reconstruction = self.loss(fx, X)
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1.0 - logsigma)/float(X.shape[0])
        return loss_reconstruction + self.beta*kl_divergence, loss_reconstruction, kl_divergence

    # inputs: x_t, x_{t+1}
    # we allow x_next to be None in case we want to infer ys for x_t in test time
    def forward(self, x_t, x_next=None):
        mu_x_t, logvar_x_t = self.encoder(x_t)
        if x_next is None:
            return None, None, None, None, None, mu_x_t
        mu_x_next, logvar_x_next = self.encoder(x_next)
        z_t, mu_z, logvar_z = self._zrepresentation(logvar_x_t, logvar_x_next, mu_x_t, mu_x_next)
        xz_t = torch.cat((x_t, z_t), dim=-1)
        x_dec = self.decoder(xz_t) # x_{t+1}
        return x_dec, x_next, mu_z, logvar_z, z_t, mu_x_t