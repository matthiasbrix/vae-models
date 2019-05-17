import os
import time

import torch
import torch.utils.data
import torchvision.utils

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO ...
LOSSES = ["epochs", "train_loss_acc", "recon_loss_acc", "kl_diverg_acc"]
Z_STATS = ["mu_z", "std_z", "varmu_z", "expected_var_z"]

class BatchMetrics():
    def __init__(self):
        self.train_loss_acc, self.recon_loss_acc, self.kl_diverg_acc, \
        self.mu_z, self.std_z, self.varmu_z, self.expected_var_z = 0.0, 0.0, 0.0, \
            0.0, 0.0, 0.0, 0.0

    def compute_train_metrics(self, train_loss, reconstruction_loss, kl_divergence, latent_space, mu_x, logvar_x):
        self.train_loss_acc += train_loss
        # accum. reconstruction loss and kl divergence
        self.recon_loss_acc += reconstruction_loss.item()
        self.kl_diverg_acc += kl_divergence.item()
        # compute mu(q(z|x)), std(q(z|x))
        self.mu_z += torch.mean(latent_space).item() # need the metric just for one batch, actually don't need for all
        self.std_z += torch.std(latent_space).item()
        # Var(mu(x))
        muzdim = torch.mean(mu_x, 0, True)
        muzdim = torch.mean(muzdim.pow(2)) # is E[\mu(x)^2]
        varmu = torch.mean(mu_x.pow(2)) # is \bar{\mu}^T\bar{\mu}
        self.varmu_z += (varmu - muzdim).item() # E[||\mu(x) - \bar{\mu}||^2]
        self.expected_var_z += torch.mean(torch.exp(logvar_x)) # E[var(q(z|x))]

class Solver(object):
    def __init__(self, model, data_loader, optimizer, z_dim, epochs, step_lr, step_config, optim_config, warmup_epochs,\
            beta, cvae_mode=False, tdcvae_mode=False):
        self.data_loader = data_loader
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.device = device

        self.folder_prefix = "../results/"
        self.save_model_dir = self.folder_prefix+self.data_loader.path+"/saved_models/"
        self.step_lr = step_lr
        self.z_dim = z_dim
        self.epochs = epochs
        self.step_config = step_config
        self.train_loss_history = {x: [] for x in LOSSES}
        self.test_loss_history = []
        self.z_stats_history = {x: [] for x in Z_STATS}
        self.labels = np.zeros((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size) # TODO: does this work? Check with plot?
        self.latent_space = np.zeros(((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size, z_dim))
        self.cvae_mode = cvae_mode
        self.tdcvae_mode = tdcvae_mode
        self.warmup_epochs = warmup_epochs
        self.beta_param = beta
        self.beta = self.beta_param if not(self.warmup_epochs) else 0

    def _save_train_metrics(self, epoch, metrics):
        num_train_samples = self.data_loader.num_train_samples
        num_train_batches = self.data_loader.num_train_batches
        train_loss = metrics.train_loss_acc/num_train_samples
        self.train_loss_history["epochs"].append(epoch) # just for debug mode (in case we finish earlier)
        self.train_loss_history["train_loss_acc"].append(train_loss)
        self.train_loss_history["recon_loss_acc"].append(metrics.recon_loss_acc/num_train_samples)
        self.train_loss_history["kl_diverg_acc"].append(metrics.kl_diverg_acc/num_train_samples)
        self.z_stats_history["mu_z"].append(metrics.mu_z/num_train_batches)
        self.z_stats_history["std_z"].append(metrics.std_z/num_train_batches)
        self.z_stats_history["varmu_z"].append(metrics.varmu_z/num_train_batches)
        self.z_stats_history["expected_var_z"].append(metrics.expected_var_z/num_train_batches)
        return train_loss

    def _train_batch(self, x, metrics, y=None):
        self.optimizer.zero_grad()
        if self.cvae_mode:
            decoded, mu_x, logvar_x, latent_space = self.model(x, y)
        elif self.tdcvae_mode:
            decoded, x, mu_x, logvar_x, latent_space = self.model(x)
        else:
            decoded, mu_x, logvar_x, latent_space = self.model(x) # vae
        loss, reconstruction_loss, kl_divergence = self.model.loss_function(decoded, x, logvar_x, mu_x, self.beta)
        loss.backward() # compute gradients
        self.optimizer.step()
        train_loss = loss.item()
        metrics.compute_train_metrics(train_loss, reconstruction_loss, kl_divergence, latent_space, mu_x, logvar_x)
        return latent_space

    def _test_batch(self, x, batch_idx, epoch, y=None):
        if self.cvae_mode:
            decoded, mu_x, logvar_x, _ = self.model(x, y)
        elif self.tdcvae_mode:
            decoded, x, mu_x, logvar_x, _ = self.model(x)
        else:
            decoded, mu_x, logvar_x, _ = self.model(x) # vae
        loss, _, _ = self.model.loss_function(decoded, x, mu_x, logvar_x, self.beta)
        if batch_idx == 0: # check w/ test set on first batch in test set.
            n = min(x.size(0), 16) # 2 x 8 grid
            comparison = torch.cat([truth.view(self.data_loader.batch_size, 1, *self.data_loader.img_dims)[:n], \
            decoded.view(self.data_loader.batch_size, 1, *self.data_loader.img_dims)[:n]])
            torchvision.utils.save_image(comparison.cpu(), self.folder_prefix + self.data_loader.folder_name \
                + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=n)
        return loss.item()

    def _train_non_labels(self, epoch):
        batch_metrics = BatchMetrics()
        self.model.train()
        for batch_idx, data in enumerate(self.data_loader.train_loader):
            x = data.to(device)
            latent_space = self._train_batch(x, batch_metrics)
            if epoch == self.epochs and batch_idx != (len(self.data_loader.train_loader)-1):
                start = batch_idx*x.size(0)
                end = (batch_idx+1)*x.size(0)
                self.latent_space[start:end, :] = latent_space.cpu().detach().numpy()
        train_loss_res = self._save_train_metrics(epoch, batch_metrics)
        print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss_res))

    def _train(self, epoch):
        batch_metrics = BatchMetrics()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            x, y = data.to(device), target.to(device)
            latent_space = self._train_batch(x, batch_metrics, y)
            if epoch == self.epochs and batch_idx != (len(self.data_loader.train_loader)-1):
                start = batch_idx*x.size(0)
                end = (batch_idx+1)*x.size(0)
                self.latent_space[start:end, :] = latent_space.cpu().detach().numpy()
        train_loss = self._save_train_metrics(epoch, batch_metrics)
        print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss))

    def _test_non_labels(self, epoch):
        test_loss_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.test_loader):
                x = data.to(device)
                test_loss_acc += self._test_batch(x, batch_idx, epoch)
        test_loss_acc /= self.data_loader.num_test_samples
        self.test_loss_history.append(test_loss_acc)
        print("====> Test set loss avg: {:.4f}".format(test_loss_acc))

    def _test(self, epoch):
        test_loss_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader.test_loader):
                x, y = data.to(device), target.to(device)
                test_loss_acc += self._test_batch(x, batch_idx, epoch, y) # TODO: Why no work?
        test_loss_acc /= self.data_loader.num_test_samples
        self.test_loss_history.append(test_loss_acc)
        print("====> Test set loss avg: {:.4f}".format(test_loss_acc))
    
    # generating samples from only the decoder
    def _sample(self, epoch):
        return # TODO
        with torch.no_grad():
            if self.cvae_mode:
                z_sample = torch.randn(100, self.z_dim).to(device) # 100 = 10 x 10 grid
                idx = torch.randint(0, self.data_loader.n_classes, (1,)).item()
                y_sample = torch.FloatTensor(torch.zeros(z_sample.size(0), self.data_loader.n_classes)) # 100 x num_classes
                y_sample[:, idx] = 1.
                sample = torch.cat((z_sample, y_sample), dim=-1)
            elif self.tdcvae_mode:
                # TODO: HOW TO HANDLE THE SAMPLING IN TDCVAE? Take a random image x, and then decode with a random z
                pass
                #z_sample = torch.randn(100, self.z_dim).to(device) # 100 = 10 x 10 grid
                #xz_t = torch.cat((x_t, z_sample), dim=-1)
                #x_t = self.decoder(xz_t) # x_{t+1}
            else:
                sample = torch.randn(100, self.z_dim).to(device) # 100 = 10 x 10 grid'''
            sample = self.model.decoder(sample)
            torchvision.utils.save_image(sample.view(100, 1, *self.data_loader.img_dims), self.folder_prefix + self.data_loader.folder_name \
                + "/generated_sample_" + str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=10)

    def _prepare_directories(self):
        os.makedirs(self.folder_prefix, exist_ok=True)
        os.makedirs(self.folder_prefix+self.data_loader.folder_name, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)

    def _save_model_params_to_file(self):
        with open(self.folder_prefix + self.data_loader.folder_name + "/model_params_" +\
            self.data_loader.dataset + "_z=" + str(self.z_dim) + ".txt", 'w') as param_file:
            param_file.write("epochs: {}\nbeta: {}\nbeta_param: {}\nwarmup_epochs: {}\ndim(z): {}\nstep_lr: {}\nbatch_size: {}\n\
                step_size: {}\ngamma: {}".format(self.epochs, self.beta, self.beta, self.warmup_epochs, self.z_dim, self.step_lr,\
                     self.data_loader.batch_size, self.step_config["step_size"], self.step_config["gamma"]))

    def main(self):
        print("+++++ START RUN +++++")
        self._prepare_directories()
        self._save_model_params_to_file()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.step_config)
        for epoch in range(1, self.epochs+1):
            t0 = time.time()
            if self.data_loader.dataset == "FF":
                self._train_non_labels(epoch)
                self._test_non_labels(epoch)
            else:
                self._train(epoch)
                self._test(epoch)
            self._sample(epoch)
            if self.step_lr:
                scheduler.step()
            if self.warmup_epochs and self.beta < self.beta_param:
                self.beta += self.warmup_epochs/self.epochs * self.beta_param
            print("{} seconds for epoch {}".format(time.time() - t0, epoch))
        print("+++++ RUN IS FINISHED +++++")
