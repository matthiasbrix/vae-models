import os
import time

import torch
import torch.utils.data
import torchvision.utils

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Solver(object):
    def __init__(self, model, data_loader, optimizer, z_dim, img_dims, epochs, num_normal_plots, batch_size=128, learning_rate=1e-3):
        self.loader = data_loader
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate) # params is iterable of parameters to optimize or dicts defining parameter groups
        self.model.to(device)
        
        self.img_dims = img_dims
        self.z_dim = z_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_loss_history = []
        self.test_loss_history = []
        self.batch_size = batch_size
        self.normal_plot_iter = epochs//num_normal_plots
        self.train_loader = self.loader.train_loader
        self.test_loader = self.loader.test_loader
        self.store_z_stats = np.arange(self.normal_plot_iter, epochs+1, self.normal_plot_iter)
        self.z_stats = []
        self.labels = np.zeros((len(self.train_loader)-1)*self.batch_size)
        self.latent_space = np.zeros(((len(self.train_loader)-1)*self.batch_size, z_dim))
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)
        self.folder_prefix = "../testing/"

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        store_z_stats = epoch in self.store_z_stats
        mu_z, std_z, rl, kl = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            X = data.to(device)
            self.optimizer.zero_grad()
            decoded, mu_x, logsigma, latent_space = self.model(X)
            loss, reconstruction_loss, kl_divergence = self.model.loss_function(decoded, X, logsigma, mu_x)
            loss.backward() # compute gradients
            train_loss += loss.item()
            self.optimizer.step()
            # accum. reconstruction loss and kl divergence
            rl += reconstruction_loss
            kl += kl_divergence
            if store_z_stats:
                # compute mu(z), std(z)
                mu_z += torch.mean(latent_space).item() # TODO need the metric just for one batch, actually don't need for all
                std_z += torch.std(latent_space).item()
                # for Var(mu(x)), just first batch
                if batch_idx == 0:
                    muzdim = torch.mean(mu_x, 0, True)/self.batch_size
                    muzdim = torch.mean(muzdim.pow(2))/self.z_dim # is E[\mu(x)^2]
                    varmu = torch.mean(mu_x.pow(2)) # is \bar{\mu}^T\bar{\mu}
                    # res measures how much overlap, because if too much, we just get an average output image, but if too low, we might just get perfect output and very little variation 
                    res = torch.abs(muzdim - varmu).item() # E[||\mu(x) - \bar{\mu}||^2] - should increase in start but then go down? if res 0, we have no variance which means no guarantee to sample a digit within the space close to it
                    # print(torch.mean(torch.exp(logsigma)))
            if epoch == self.epochs and batch_idx != (len(self.train_loader)-1):
                # store the latent space of all digits in last epoch
                start = batch_idx*data.shape[0]
                end = (batch_idx+1)*data.shape[0]
                self.labels[start:end] = target.numpy()
                self.latent_space[start:end, :] = latent_space.detach().numpy()
            #if batch_idx % 10 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(X), len(self.train_loader.dataset),
            #        100. * batch_idx / len(self.train_loader),
            #        loss.item() / len(X)))
        train_loss /= self.num_train_samples
        rl /= self.num_train_samples
        kl /= self.num_train_samples
        self.train_loss_history.append((epoch, train_loss, rl, kl))
        #print("both mu", mu_z/self.num_train_batches, torch.mean(mu_x))
        #print((std_z/self.num_train_batches)**2, torch.mean(torch.exp(1/2*logsigma)))
        if store_z_stats:
            self.z_stats.append((epoch, mu_z/self.num_train_batches, std_z/self.num_train_batches, res))
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
                    n = min(data.size(0), 16) # TODO: 16 should be a hyperparam
                    comparison = torch.cat([data[:n], decoded.view(self.batch_size, 1, *self.img_dims)[:n]])
                    torchvision.utils.save_image(comparison.cpu(), self.folder_prefix + self.loader.folder_name + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=n)        
        test_loss /= len(self.test_loader.dataset)
        print("====> Test set loss avg: {:.4f}".format(test_loss))
        self.test_loss_history.append(test_loss)
        return test_loss
    
    def run(self):
        os.makedirs(self.folder_prefix, exist_ok=True)
        os.makedirs(self.folder_prefix+self.loader.folder_name, exist_ok=True)
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        print("+++++ START RUN +++++")
        for epoch in range(1, self.epochs+1):
            #scheduler.step()
            t0 = time.time()
            self.train(epoch)
            tl = self.test(epoch)
            scheduler2.step(tl)
            with torch.no_grad():
                # In test time we disregard the encoder and only generate z from N(0,I) which we use as arg to decoder
                sample = torch.randn(64, self.z_dim).to(device) # TODO 64 should be a hyperparam 
                sample = self.model.decoder(sample)
                torchvision.utils.save_image(sample.view(64, 1, *self.img_dims), self.folder_prefix + self.loader.folder_name + "/test_sample_" + str(epoch) + "_z=" + str(self.z_dim) + ".png") # inserting a mini batch tensor to compute a grid
            print('{} seconds for epoch {}'.format(time.time() - t0, epoch))
        print("+++++ RUN IS FINISHED +++++")
