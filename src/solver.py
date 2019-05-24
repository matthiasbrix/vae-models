import time

import torch
import torch.utils.data
import torchvision.utils
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EpochMetrics():
    def __init__(self):
        self.train_loss_acc, self.test_loss_acc, self.recon_loss_acc, self.kl_diverg_acc,\
        self.mu_z, self.std_z, self.varmu_z, self.expected_var_z = 0.0, 0.0, 0.0, 0.0,\
            0.0, 0.0, 0.0, 0.0

    def compute_batch_train_metrics(self, train_loss, reconstruction_loss, kl_divergence,\
        latent_space, mu_x, logvar_x):
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

    def compute_batch_test_metrics(self, test_loss):
        self.test_loss_acc += test_loss

class Training(object):
    def __init__(self, solver):
        self.solver = solver
     
    def _train_batch(self, epoch_metrics, x, y=None):
        y_space = None
        theta_diff = None
        self.solver.optimizer.zero_grad()
        if self.solver.cvae_mode:
            x = x.view(-1, self.solver.data_loader.input_dim)
            decoded, mu_x, logvar_x, latent_space = self.solver.model(x, y)
        elif self.solver.tdcvae_mode:
            x_rot, x_next, theta_diff, theta_1 = self.solver.prepro.preprocess_batch(x, self.solver.data_loader.input_dim)
            x_rot, x_next = x_rot.to(self.solver.device), x_next.to(self.solver.device) # TODO: do it the line before
            decoded, x, mu_x, logvar_x, latent_space, y_space = self.solver.model(x_rot, x_next)
        else:
            x = x.view(-1, self.solver.data_loader.input_dim)
            decoded, mu_x, logvar_x, latent_space = self.solver.model(x) # vae
        loss, reconstruction_loss, kl_divergence = \
            self.solver.model.loss_function(decoded, x, logvar_x, mu_x, self.solver.beta)
        loss.backward() # compute gradients
        self.solver.optimizer.step()
        epoch_metrics.compute_batch_train_metrics(loss.item(), reconstruction_loss,\
            kl_divergence, latent_space, mu_x, logvar_x)
        return latent_space, y_space, theta_diff, theta_1

    def train(self, epoch, epoch_metrics):
        self.solver.model.train()
        for batch_idx, data in enumerate(self.solver.data_loader.train_loader):
            if self.solver.data_loader.with_labels:
                x, y = data[0].to(self.solver.device), data[1].to(self.solver.device) # TODO: don't do if tdcvae because of prepro, because otherwise CPU -> GPU -> CPU -> GPU
                latent_space, y_space, theta_diff, theta_1 = self._train_batch(epoch_metrics, x, y)
            else:
                x = data.to(self.solver.device)
                latent_space, y_space, theta_diff, theta_1 = self._train_batch(epoch_metrics, x)
            if epoch == self.solver.epochs and batch_idx != (len(self.solver.data_loader.train_loader)-1):
                start = batch_idx*x.size(0)
                end = (batch_idx+1)*x.size(0)
                self.solver.latent_space[start:end, :] = latent_space.cpu().detach().numpy()
                if y is not None and not self.solver.tdcvae_mode:
                    self.solver.labels[start:end] = y.cpu().detach().numpy()
                if self.solver.tdcvae_mode and theta_diff:
                    self.solver.labels[start:end] = np.repeat(theta_diff, x.size(0))
                if self.solver.tdcvae_mode and theta_1:
                    self.solver.y_space_labels[start:end] = np.repeat(theta_1, x.size(0))
                if y_space is not None:
                    self.solver.y_space[start:end, :] = y_space.cpu().detach().numpy()

class Testing(object):
    def __init__(self, solver):
        self.solver = solver

    def _test_batch(self, epoch_metrics, batch_idx, epoch, x, y=None):
        if self.solver.cvae_mode:
            x = x.view(-1, self.solver.data_loader.input_dim)
            decoded, mu_x, logvar_x, _ = self.solver.model(x, y)
        elif self.solver.tdcvae_mode:
            x_rot, x_next, _, _ = self.solver.prepro.preprocess_batch(x, self.solver.data_loader.input_dim)
            x_rot, x_next = x_rot.to(self.solver.device), x_next.to(self.solver.device) # TODO: do it the line before
            decoded, x, mu_x, logvar_x, _, _ = self.solver.model(x_rot, x_next)
        else:
            x = x.view(-1, self.solver.data_loader.input_dim)
            decoded, mu_x, logvar_x, _ = self.solver.model(x) # vae
        loss, _, _ = self.solver.model.loss_function(decoded, x, mu_x, logvar_x, self.solver.beta)
        epoch_metrics.compute_batch_test_metrics(loss.item())
        if batch_idx == 0: # check w/ test set on first batch in test set.
            n = min(x.size(0), 16) # 2 x 8 grid
            comparison = torch.cat([x.view(self.solver.data_loader.batch_size, 1, *self.solver.data_loader.img_dims)[:n],\
            decoded.view(self.solver.data_loader.batch_size, 1, *self.solver.data_loader.img_dims)[:n]])
            torchvision.utils.save_image(comparison.cpu(), self.solver.data_loader.result_dir \
                + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.solver.z_dim) + ".png", nrow=n)

    def test(self, epoch, epoch_metrics):
        self.solver.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.solver.data_loader.test_loader):
                if self.solver.data_loader.with_labels:
                    x, y = data[0].to(self.solver.device), data[1].to(self.solver.device)
                    self._test_batch(epoch_metrics, batch_idx, epoch, x, y)
                else:
                    x = data.to(self.solver.device)
                    self._test_batch(epoch_metrics, batch_idx, epoch, x)

class Solver(object):
    def __init__(self, model, data_loader, optimizer, z_dim, epochs, beta, step_config,\
            optim_config, lr_scheduler=None, num_samples=100, cvae_mode=False,\
            tdcvae_mode=False, prepro=None):
        self.data_loader = data_loader
        self.model = model
        self.prepro = prepro
        self.model.to(DEVICE)
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.device = DEVICE

        self.z_dim = z_dim
        self.epochs = epochs
        self.beta = beta
        self.step_config = step_config
        self.lr_scheduler = lr_scheduler(self.optimizer, **step_config) if lr_scheduler else lr_scheduler
        self.train_loss_history = {x: [] for x in ["epochs", "train_loss_acc", "recon_loss_acc", "kl_diverg_acc"]}
        self.test_loss_history = []
        self.z_stats_history = {x: [] for x in ["mu_z", "std_z", "varmu_z", "expected_var_z"]}
        self.latent_space = np.zeros(((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size, z_dim))
        self.labels = np.zeros((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size) # TODO: rename to latent_space_labels
        self.y_space = np.zeros(((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size, z_dim))
        self.y_space_labels = np.zeros((len(self.data_loader.train_loader)-1)*self.data_loader.batch_size)
        self.cvae_mode = cvae_mode
        self.tdcvae_mode = tdcvae_mode
        self.num_samples = num_samples

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
    
    def _save_test_metrics(self, metrics):
        test_loss = metrics.test_loss_acc/self.data_loader.num_test_samples
        self.test_loss_history.append(test_loss)
        return test_loss

    # generating samples from only the decoder
    def _sample(self, epoch, num_samples):
        with torch.no_grad():
            if self.cvae_mode:
                z_sample = torch.randn(num_samples, self.z_dim).to(self.device)
                idx = torch.randint(0, self.data_loader.n_classes, (1,)).item()
                y_sample = torch.FloatTensor(torch.zeros(z_sample.size(0), self.data_loader.n_classes)) # 100 x num_classes
                y_sample[:, idx] = 1.
                sample = torch.cat((z_sample, y_sample), dim=-1)
            elif self.tdcvae_mode:
                z_sample = torch.randn(num_samples, self.z_dim).to(self.device)
                x_t = iter(self.data_loader.train_loader).next()[0][:num_samples]
                x_t = x_t.view(-1, self.data_loader.input_dim).to(self.device)
                sample = torch.cat((x_t, z_sample), dim=-1)
            else:
                sample = torch.randn(num_samples, self.z_dim).to(self.device)
            sample = self.model.decoder(sample)
            torchvision.utils.save_image(sample.view(num_samples, 1, *self.data_loader.img_dims), \
                    self.data_loader.result_dir  + "/generated_sample_" +\
                    str(epoch) + "_z=" + str(self.z_dim) + ".png", nrow=10)

    def _save_model_params_to_file(self):
        with open(self.data_loader.result_dir + "/model_params_" +\
            self.data_loader.dataset + "_z=" + str(self.z_dim) + ".txt", 'w') as param_file:
            params = "epochs: {}\n"\
                "optimizer: {}\n"\
                "beta: {}\n"\
                "dim(z): {}\n"\
                "batch_size: {}\n"\
                "lr_scheduler: {}\n"\
                "step_config: {}\n"\
                .format(self.epochs, self.optimizer, self.beta, self.z_dim, self.data_loader.batch_size,\
                    self.lr_scheduler, self.step_config)
            if self.prepro:
                if self.prepro.rotate:
                    params += "thetas: (theta_1: {}, theta_2: {})\n"\
                        .format(self.prepro.theta_range_1, self.prepro.theta_range_2)
                if self.prepro.scale:
                    params += "scales: {}\n"\
                        .format(self.prepro.scale_range_1)
                params += str(self.model)
            param_file.write(params)

    def main(self):
        print("+++++ START RUN +++++")
        self._save_model_params_to_file()
        training = Training(self)
        testing = Testing(self)
        for epoch in range(1, self.epochs+1):
            epoch_watch = time.time()
            epoch_metrics = EpochMetrics()
            training.train(epoch, epoch_metrics)
            train_loss = self._save_train_metrics(epoch, epoch_metrics)
            print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss))
            testing.test(epoch, epoch_metrics)
            test_loss = self._save_test_metrics(epoch_metrics)
            print("====> Test set loss avg: {:.4f}".format(test_loss))
            self._sample(epoch, self.num_samples)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            print("{:.2f} seconds for epoch {}".format(time.time() - epoch_watch, epoch))
        print("+++++ RUN IS FINISHED +++++")
