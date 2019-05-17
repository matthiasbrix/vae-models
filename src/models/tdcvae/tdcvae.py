from pathlib import Path

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PATH = str(Path(__file__).parent.absolute()).split('/')[-1]

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

'''prepro = Prepro()
        for batch_idx, data in enumerate(data_loader.train_loader):
            #asd = prepro.rand_rotate(data[0][0], angles=[0, 360])
            #asd = prepro.det_rotate(data[0][0], angle=90)
            plt.imshow(asd.view(28, 28).numpy())
            break
        without rotate and only decode(z_t): works ok, but test loss is 2x as train loss.
        without rotate but with decode(xz_t):  works quite well.
        with rotate + decode(xz_t):
'''

class Prepro():

    def __init__(self):
        pass

    def rand_rotate(self, img, angles):
        transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(angles),
                transforms.ToTensor()
            ])
        return transform_img(img)

    def det_rotate(self, img, angle):
        img = TF.rotate(transforms.ToPILImage()(img), angle)
        return transforms.ToTensor()(img)

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(x)
        self.linear22 = nn.Linear(H, Dout) # \Sigma(x)
        self.batch_norm = nn.BatchNorm1d(H)
        self.relu = nn.ReLU()

    # compute \mu(x_t), \sigma(x_t), so p(y_t|x_t)
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

    # compute p(x_{t+1}|x_t, z_{t+1})
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.sigmoid(x)

class TD_Cvae(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, T):
        super(TD_Cvae, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim+input_dim, hidden_dim, input_dim)
        self.T = T
        self.z_dim = z_dim
        self.prepro = Prepro() # TODO put prepro in different file

    # y_t \sim N(\mu(x_t), \sigma(x_t))
    def _reparameterization_trick(self, mu_x_t, logvar_x_t):
        sigma_t = torch.exp(1/2*logvar_x_t)
        eps = torch.randn_like(sigma_t)
        return mu_x_t + sigma_t*eps

    # z_t = (y_t - y_{t-1}) \sim N(\mu(x_t)-\mu(x_t), \Sigma(x_t)+\Sigma(x_{t-1}))
    def _zrepresentation(self, y_t, y_prev):
        return y_t - y_prev

    # loss function + KL divergence, use for this \mu(x), \Sigma(x)
    # compute here D_{KL}[N(\mu(x), \Sigma(x))||N(0,1)]
    def loss_function(self, fx, X, logsigma, mu, beta):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum")
        kl_divergence = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma)
        return loss_reconstruction + beta*kl_divergence, loss_reconstruction, beta*kl_divergence

    # TODO: take two inputs...
    def forward(self, x_t):
        theta_1 = 0 #np.random.randint(-180, 181)
        theta_2 = theta_1 + np.random.randint(-30, 31)
        x_t = x_t.view(-1, self.input_dim) # TODO: can't take a batch.... should fix the roatet
        x_rot = self.prepro.det_rotate(x_t, theta_1).view(-1, self.input_dim)
        x_next = self.prepro.det_rotate(x_t, theta_2).view(-1, self.input_dim)
        #y_= torch.zeros((x_t.size(0), self.z_dim)) # y_0
        mu_x_t, logvar_x_t = self.encoder(x_rot)
        mu_x_next, logvar_x_next = self.encoder(x_next)
        y_t = self._reparameterization_trick(mu_x_t, logvar_x_t)
        y_next = self._reparameterization_trick(mu_x_next, logvar_x_next)
        z_t = self._zrepresentation(y_next, y_t)#.view(1, 128, 20)
        # If z_t = 0 then we skip the decoding and just return the input
        xz_t = torch.cat((x_rot, z_t), dim=-1)
        x_dec = self.decoder(xz_t) # x_{t+1}
        #x_t = self.prepro.det_rotate(x_t, theta_2).view(-1, self.input_dim)
        return x_dec, x_next, mu_x_t-mu_x_next, torch.log(torch.exp(logvar_x_t)+torch.exp(logvar_x_t)), z_t