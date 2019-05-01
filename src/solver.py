import os
import time

import torch
import torch.utils.data
import torchvision.utils

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
losses = ["epochs", "train_loss_acc", "recon_loss_acc", "kl_diverg_acc"]
z_stats = ["mu_z", "std_z", "varmu_z", "expected_var_z"]

class Solver(object):
    def __init__(self, model, data_loader, optimizer, z_dim, epochs, step_config, optim_config, batch_size=128):
        self.loader = data_loader
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.device = device

        self.z_dim = z_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = self.loader.train_loader
        self.test_loader = self.loader.test_loader
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)
        self.folder_prefix = "../results/"
        self.step_config = step_config
        self.train_loss_history = {x: [] for x in losses}
        self.test_loss_history = []
        self.z_stats_history = {x: [] for x in z_stats}
        self.labels = np.zeros((len(self.train_loader)-1)*self.batch_size)
        self.latent_space = np.zeros(((len(self.train_loader)-1)*self.batch_size, z_dim))

    def train_non_labels(self, epoch):
        train_loss_acc, recon_loss_acc, kl_diverg_acc, \
            mu_z, std_z, varmu_z, expected_var_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            X = data.to(device)
            self.optimizer.zero_grad()
            decoded, mu_x, logvar_x, latent_space = self.model(X)
            loss, reconstruction_loss, kl_divergence = self.model.loss_function(decoded, X, logvar_x, mu_x)
            loss.backward() # compute gradients
            train_loss_acc += loss.item()
            self.optimizer.step()
            # accum. reconstruction loss and kl divergence
            recon_loss_acc += reconstruction_loss.item()
            kl_diverg_acc += kl_divergence.item()
            # compute mu(q(z|x)), std(q(z|x))
            mu_z += torch.mean(latent_space).item() # need the metric just for one batch, actually don't need for all
            std_z += torch.std(latent_space).item()
            # Var(mu(x))
            muzdim = torch.mean(mu_x, 0, True)
            muzdim = torch.mean(muzdim.pow(2)) # is E[\mu(x)^2]
            varmu = torch.mean(mu_x.pow(2)) # is \bar{\mu}^T\bar{\mu}
            varmu_z += (varmu - muzdim).item() # E[||\mu(x) - \bar{\mu}||^2]
            expected_var_z += torch.mean(torch.exp(logvar_x).pow(2)) # E[var(q(z|x))]
            if epoch == self.epochs and batch_idx != (len(self.train_loader)-1):
                # store the latent space of all digits in last epoch
                start = batch_idx*X.shape[0]
                end = (batch_idx+1)*X.shape[0]
                self.latent_space[start:end, :] = latent_space.cpu().detach().numpy()
            # if batch_idx % 10 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(X), len(self.train_loader.dataset),
            #        100. * batch_idx / len(self.train_loader),
            #        loss.item() / len(X)))
        self.train_loss_history["epochs"].append(epoch) # just for debug
        self.train_loss_history["train_loss_acc"].append(train_loss_acc/self.num_train_samples)
        self.train_loss_history["recon_loss_acc"].append(recon_loss_acc/self.num_train_samples)
        self.train_loss_history["kl_diverg_acc"].append(kl_diverg_acc/self.num_train_samples)
        self.z_stats_history["mu_z"].append(mu_z/self.num_train_batches)
        self.z_stats_history["std_z"].append(std_z/self.num_train_batches)
        self.z_stats_history["varmu_z"].append(varmu_z/self.num_train_batches)
        self.z_stats_history["expected_var_z"].append(expected_var_z/self.num_train_batches)
        print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss_acc/self.num_train_samples))

    def train(self, epoch):
        train_loss_acc, recon_loss_acc, kl_diverg_acc, \
            mu_z, std_z, varmu_z, expected_var_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            X = data.to(device)
            self.optimizer.zero_grad()
            decoded, mu_x, logvar_x, latent_space = self.model(X)
            loss, reconstruction_loss, kl_divergence = self.model.loss_function(decoded, X, logvar_x, mu_x)
            loss.backward() # compute gradients
            self.optimizer.step()
            train_loss_acc += loss.item()
            # accum. reconstruction loss and kl divergence
            recon_loss_acc += reconstruction_loss.item()
            kl_diverg_acc += kl_divergence.item()
            # compute mu(z), std(z)
            mu_z += torch.mean(latent_space).item() # need the metric just for one batch, actually don't need for all
            std_z += torch.std(latent_space).item()
            # Var(mu(x))
            muzdim = torch.mean(mu_x, 0, True)
            muzdim = torch.mean(muzdim.pow(2)) # is E[\mu(x)^2]
            varmu = torch.mean(mu_x.pow(2)) # is \bar{\mu}^T\bar{\mu}
            varmu_z += (varmu - muzdim).item() # E[||\mu(x) - \bar{\mu}||^2]
            expected_var_z += torch.mean(torch.exp(logvar_x).pow(2)) # E[var(q(z|x))]
            if epoch == self.epochs and batch_idx != (len(self.train_loader)-1):
                # store the latent space of all digits in last epoch
                start = batch_idx*X.shape[0]
                end = (batch_idx+1)*X.shape[0]
                self.labels[start:end] = target.numpy()
                self.latent_space[start:end, :] = latent_space.cpu().detach().numpy()
            #if batch_idx % 10 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(X), len(self.train_loader.dataset),
            #        100. * batch_idx / len(self.train_loader),
            #        loss.item() / len(X)))
        self.train_loss_history["epochs"].append(epoch) # just for debug
        self.train_loss_history["train_loss_acc"].append(train_loss_acc/self.num_train_samples)
        self.train_loss_history["recon_loss_acc"].append(recon_loss_acc/self.num_train_samples)
        self.train_loss_history["kl_diverg_acc"].append(kl_diverg_acc/self.num_train_samples)
        self.z_stats_history["mu_z"].append(mu_z/self.num_train_batches)
        self.z_stats_history["std_z"].append(std_z/self.num_train_batches)
        self.z_stats_history["varmu_z"].append(varmu_z/self.num_train_batches)
        self.z_stats_history["expected_var_z"].append(expected_var_z/self.num_train_batches)
        print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss_acc/self.num_train_samples))

    def test_non_labels(self, epoch):
        test_loss_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                X = data.to(device)
                decoded, mu_x, logvar_x, _ = self.model(X)
                loss, _, _ = self.model.loss_function(decoded, X, mu_x, logvar_x)
                test_loss_acc += loss.item()
                if i == 0: # check w/ test set on first batch in test set.
                    n = min(X.size(0), 16) # 2 x 8 grid
                    comparison = torch.cat([X[:n], decoded.view(self.batch_size, 1, *self.loader.img_dims)[:n]])
                    torchvision.utils.save_image(comparison.cpu(), self.folder_prefix + self.loader.folder_name \
                    + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=n)
        test_loss_acc /= len(self.test_loader.dataset)
        self.test_loss_history.append(test_loss_acc)
        print("====> Test set loss avg: {:.4f}".format(test_loss_acc))

    def test(self, epoch):
        test_loss_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                X = data.to(device)
                decoded, mu_x, logvar_x, _ = self.model(X)
                loss, _, _ = self.model.loss_function(decoded, X, mu_x, logvar_x)
                test_loss_acc += loss.item()
                if i == 0: # check w/ test set on first batch in test set.
                    n = min(X.size(0), 16) # 2 x 8 grid
                    comparison = torch.cat([X[:n], decoded.view(self.batch_size, 1, *self.loader.img_dims)[:n]])
                    torchvision.utils.save_image(comparison.cpu(), self.folder_prefix + self.loader.folder_name \
                    + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=n)        
        test_loss_acc /= len(self.test_loader.dataset)
        self.test_loss_history.append(test_loss_acc)
        print("====> Test set loss avg: {:.4f}".format(test_loss_acc))
    
    def run(self):
        os.makedirs(self.folder_prefix, exist_ok=True)
        os.makedirs("../models/", exist_ok=True)
        os.makedirs(self.folder_prefix+self.loader.folder_name, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.step_config)
        print("+++++ START RUN +++++")
        for epoch in range(1, self.epochs+1):
            t0 = time.time()
            if self.loader.dataset == "LFW":
                scheduler.step()
            if self.loader.dataset == "FF":
                self.train_non_labels(epoch)
                self.test_non_labels(epoch)
            else:
                self.train(epoch)
                self.test(epoch)
            with torch.no_grad():
                # In test time we disregard the encoder and only generate z from N(0,I) which we use as arg to decoder
                sample = torch.randn(100, self.z_dim).to(device) # 100 = 10 x 10 grid
                sample = self.model.decoder(sample)
                torchvision.utils.save_image(sample.view(100, 1, *self.loader.img_dims), self.folder_prefix + self.loader.folder_name \
                    + "/generated_sample_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=10)
            print("{} seconds for epoch {}".format(time.time() - t0, epoch))
        print("+++++ RUN IS FINISHED +++++")
