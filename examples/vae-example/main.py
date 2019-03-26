import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import os
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################
# Alot of code is copied from
# https://github.com/pytorch/examples/blob/master/vae/main.py
#########################################################################

class DataLoader():
    def __init__(self, batch_size):
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        mnist_train = datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=True)
        mnist_test = datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True, **kwargs)

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(X)
        self.linear22 = nn.Linear(H, Dout) # \sum(X)

    # compute q(z|x) which is encoding X into z
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear21(x), self.linear22(x) # \mu(X), \sum(X) so mean(X) and covariance(X)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.sigmoid = nn.Sigmoid()

    # compute P(X|z) (posterior) which is decoding to reconstruct X
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.sigmoid(self.linear2(x))

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # sampling from N(\mu(X), \sum(X))
    def reparameterization_trick(self, mu, logsigma):
        sigma = torch.exp(1/2*logsigma)
        eps = torch.randn_like(sigma) # sampling eps ~ N(0, I)
        return mu + sigma*eps # compute z = \mu(X) + \sum^{1/2}(X) * eps

    # loss function + KL divergence, use for this \mu(X), \sum(X)
    # compute here D_{KL}[N(\mu(X), \sum(X))||N(0,1)] = 1/2 \sum_k (\sum(X)+\mu^2(X) - 1 - log \sum(X))
    def loss_function(self, fx, X, logsigma, mu):
        loss_reconstruction = F.binary_cross_entropy(fx, X, reduction="sum") # E[log p(x|z)] TODO: why sum reduction?
        kl_divergence = 1/2 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) # by appendix B in the Auto Encoding Variational Bayes
        #kl_divergence2 = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma) # will give same value but negative would need to + below
        return loss_reconstruction - kl_divergence, loss_reconstruction, -kl_divergence

    def forward(self, data):
        mu, logsigma = self.encoder(data.view(-1, 784))
        z = self.reparameterization_trick(mu, logsigma)
        decoded = self.decoder(z)
        return decoded, mu, logsigma, z

class Solver(object):
    def __init__(self, optimizer, input_dim, hidden_dim, z_dim, epochs, num_normal_plots, batch_size=128, learning_rate=1e-3):
        encoder = Encoder(input_dim, hidden_dim, z_dim)
        decoder = Decoder(z_dim, hidden_dim, input_dim)
        loader = DataLoader(batch_size)
        self.model = VAE(encoder, decoder)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate) # params is iterable of parameters to optimize or dicts defining parameter groups
        self.z_dim = z_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_loss_history = []
        self.test_loss_history = []
        self.model.to(device)
        self.batch_size = batch_size
        self.normal_plot_iter = epochs//num_normal_plots
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.store_z_stats = np.arange(self.normal_plot_iter, epochs+1, self.normal_plot_iter)
        self.z_stats = []
        self.labels = np.zeros((len(self.train_loader)-1)*self.batch_size)
        self.latent_space = np.zeros(((len(self.train_loader)-1)*self.batch_size, z_dim))
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        store_z_stats = epoch in self.store_z_stats
        mu_z, std_z, rl_z, kl_z = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            X = data.to(device)
            self.optimizer.zero_grad()
            decoded, mu, logsigma, latent_space = self.model(X) # shapes are torch.Size([128, 784]) torch.Size([128, 20]) torch.Size([128, 20]) because batch_size = 128
            loss, reconstruction_loss, kl_divergence = self.model.loss_function(decoded, X, logsigma, mu)
            loss.backward() # compute gradients
            train_loss += loss.item()
            self.optimizer.step()
            if store_z_stats:
                # compute mean(z), std(z) and accumulate it
                mu_z += torch.mean(latent_space).item()
                std_z += torch.std(latent_space).item()
            if epoch == self.epochs and batch_idx != (len(self.train_loader)-1):
                # store the latent space of all digits in last epoch
                start = batch_idx*data.shape[0]
                end = (batch_idx+1)*data.shape[0]
                self.labels[start:end] = target.numpy()
                self.latent_space[start:end, :] = latent_space.detach().numpy()
            # accum. reconstruction loss and kl divergence
            rl_z += reconstruction_loss
            kl_z += kl_divergence
            #if batch_idx % 10 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(X), len(self.train_loader.dataset),
            #        100. * batch_idx / len(self.train_loader),
            #        loss.item() / len(X)))
        train_loss /= self.num_train_samples # TODO: Divide by number of batches?
        rl_z /= self.num_train_samples
        kl_z /= self.num_train_samples
        self.train_loss_history.append((epoch, train_loss, rl_z, kl_z))
        if store_z_stats:
            self.z_stats.append((epoch, mu_z/self.num_train_batches, std_z/self.num_train_batches))
        print("====> Epoch: {} train set loss avg: {:.4f}".format(
            epoch, train_loss))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(device)
                decoded, mu, logvar, _ = self.model(data)
                loss, _, _ = self.model.loss_function(decoded, data, mu, logvar)
                test_loss += loss.item()
                if i == 0: # check w/ test set on first batch in test set.
                    n = min(data.size(0), 16)
                    comparison = torch.cat([data[:n], decoded.view(self.batch_size, 1, 28, 28)[:n]])
                    torchvision.utils.save_image(comparison.cpu(), "testing/test_reconstruction_" + str(epoch) + ".png", nrow=n)
        test_loss /= len(self.test_loader.dataset)
        print("====> Test set loss avg: {:.4f}".format(test_loss))
        self.test_loss_history.append(test_loss)
    
    def run(self):
        os.makedirs("testing", exist_ok=True)
        #scheduler = optim.lr_scheduler.StepLR(optimalg, step_size=1000, gamma=0.1)
        print("+++++ START RUN +++++")
        for epoch in range(1, self.epochs+1):
            t0 = time.time()
            #scheduler.step()
            self.train(epoch)
            self.test(epoch)
            with torch.no_grad():
                # In test time we disregard the encoder and only generate z from N(0,I) which we use as arg to decoder
                sample = torch.randn(64, self.z_dim).to(device)
                sample = self.model.decoder(sample)
                torchvision.utils.save_image(sample.view(64, 1, 28, 28), "testing/test_sample_" + str(epoch) + "_z=" + str(self.z_dim) + ".png") # inserting a mini batch tensor to compute a grid
            print('{} seconds for epoch {}'.format(time.time() - t0, epoch))
        print("+++++ RUN IS FINISHED +++++")

if __name__ == "__main__":
    input_dim = 784
    hidden_dim = 500 # Kingma, Welling use 500 neurons, otherwise use 400
    z_dim = 20 # 1000 is suggested in the paper "Tutorial on VAE" but Kingma, Welling show 20 is sufficient for MNIST
    optimizer = torch.optim.Adam
    epochs = 10000
    num_normal_plots = 2
    solver = Solver(optimizer, input_dim, hidden_dim, z_dim, epochs, num_normal_plots)
    solver.run()