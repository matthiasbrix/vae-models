"""This module is a wrapper around the training/testing procedures
with entities for saving the metrics during the procedures and sampling.

"""

import time
import argparse
import os

import torch
import torch.utils.data
import torchvision.utils

from models.vae.vae import Vae
from models.cvae.cvae import Cvae
from models.tdcvae.tdcvae import TD_Cvae
from models.tdcvae2.tdcvae2 import Tdcvae2
from model_params import get_model_data_vae, get_model_data_cvae,\
    get_model_data_tdcvae, get_model_data_tdcvae2
from directories import Directories
from dataloader import DataLoader
from sampling import vae_sampling, cvae_sampling, tdcvae_sampling, tdcvae2_sampling

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EpochMetrics():
    """Metrics of each epoch that are stored."""
    def __init__(self):
        self.train_loss_acc, self.test_loss_acc, self.recon_loss_acc, self.kl_diverg_acc,\
        self.mu_z, self.std_z, self.varmu_z, self.expected_var_z = 0.0, 0.0, 0.0, 0.0,\
            0.0, 0.0, 0.0, 0.0

    def compute_batch_train_metrics(self, train_loss, reconstruction_loss, kl_divergence,\
        z_space, mu_x, logvar_x):
        """Computes and accumulates the metrics for the training. All params are based on a batch.

        Args:
            train_loss: loss of the training of the batch, sum of reconstruction loss and kl divergence
            reconstruction_loss: the loss for reconstruction
            kl_divergence: kl divergence of q(z|x)||p(z)
            z_space: the encoded z latent space
            mu_x: encoded mu of x
            logvar_x: encoded log variance of x
        """
        self.train_loss_acc += train_loss.item()
        # accum. reconstruction loss and kl divergence
        self.recon_loss_acc += reconstruction_loss.item()
        self.kl_diverg_acc += kl_divergence.item()
        # compute mu(q(z)), std(q(z))
        self.mu_z += torch.mean(z_space).item() # need the metric just for one batch
        self.std_z += torch.std(z_space).item()
        # Var(mu(x))
        muzdim = torch.mean(mu_x, 0, True)
        muzdim = torch.mean(muzdim).pow(2) # is \bar{\mu}^T\bar{\mu}
        varmu = torch.mean(mu_x.pow(2)) # is E[\mu(x)]^2
        self.varmu_z += (varmu - muzdim).item() # E[||\mu(x) - \bar{\mu}||^2]
        self.expected_var_z += torch.mean(torch.exp(logvar_x)) # E[var(q(z|x))]

    def compute_batch_test_metrics(self, test_loss):
        """Computes and accumulates the metrics for the testing.

            Args:
                test_loss: test loss during test time of a batch
        """
        self.test_loss_acc += test_loss

class Training(object):
    """Class for training a model"""
    def __init__(self, solver):
        self.solver = solver

    def _train_batch(self, epoch_metrics, x, y=None):
        self.solver.optimizer.zero_grad()
        if self.solver.cvae_mode:
            x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            y = y.to(DEVICE)
            decoded, mu_x, logvar_x, z_space = self.solver.model(x, y)
        elif self.solver.tdcvae_mode:
            x_t, x_next = x
            x_t, x_next = x_t.view(-1, self.solver.data_loader.input_dim).to(DEVICE),\
                x_next.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            decoded, x, mu_x, logvar_x, z_space, _ = self.solver.model(x_t, x_next)
        elif self.solver.tdcvae2_mode:
            x_t, x_next = x
            x_t, x_next = x_t.to(DEVICE), x_next.to(DEVICE)
            decoded, x, mu_x, logvar_x, z_space, _ = self.solver.model(x_t, x_next)
        else:
            x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            decoded, mu_x, logvar_x, z_space = self.solver.model(x) # vae
        loss, reconstruction_loss, kl_divergence = \
            self.solver.model.loss_function(decoded, x, logvar_x, mu_x)
        loss.backward() # compute gradients
        self.solver.optimizer.step()
        epoch_metrics.compute_batch_train_metrics(loss, reconstruction_loss,\
            kl_divergence, z_space, mu_x, logvar_x)

    def train(self, epoch_metrics):
        """Method for start training a loaded dataset in the train_loader object"""
        self.solver.model.train()
        for _, train_batch in enumerate(self.solver.data_loader.train_loader):
            if self.solver.data_loader.with_labels:
                x, y = train_batch[0], train_batch[1]
                self._train_batch(epoch_metrics, x, y)
            else:
                x = train_batch
                self._train_batch(epoch_metrics, x)

class Testing(object):
    """Class for testing a model"""
    def __init__(self, solver):
        self.solver = solver

    def _test_batch(self, epoch_metrics, batch_idx, epoch, x, y=None):
        if self.solver.cvae_mode:
            x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            y = y.to(DEVICE)
            decoded, mu_x, logvar_x, _ = self.solver.model(x, y)
        elif self.solver.tdcvae_mode:
            x_t, x_next = x
            x_t, x_next = x_t.view(-1, self.solver.data_loader.input_dim).to(DEVICE),\
                x_next.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            decoded, x, mu_x, logvar_x, _, _ = self.solver.model(x_t, x_next)
        elif self.solver.tdcvae2_mode:
            x_t, x_next = x
            x_t, x_next = x_t.to(DEVICE), x_next.to(DEVICE)
            decoded, x, mu_x, logvar_x, _, _ = self.solver.model(x_t, x_next)
        else:
            x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
            decoded, mu_x, logvar_x, _ = self.solver.model(x) # vae
        loss, _, _ = self.solver.model.loss_function(decoded, x, mu_x, logvar_x)
        epoch_metrics.compute_batch_test_metrics(loss.item())
        if batch_idx == 0 and self.solver.data_loader.directories.make_dirs: # check w/ test set on first batch in test set.
            n = min(x.size(0), 16) # 2 x 8 grid
            comparison = torch.cat([x.view(x.size(0), *self.solver.data_loader.img_dims)[:n],\
                decoded.view(x.size(0), *self.solver.data_loader.img_dims)[:n]])
            torchvision.utils.save_image(comparison.cpu(), self.solver.data_loader.directories.result_dir \
                + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.solver.model.z_dim) + ".png", nrow=n)

    def test(self, epoch, epoch_metrics):
        """Method for start testing a loaded dataset in the test_loader object"""
        self.solver.model.eval()
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(self.solver.data_loader.test_loader):
                if self.solver.data_loader.with_labels:
                    x, y = test_batch[0], test_batch[1]
                    self._test_batch(epoch_metrics, batch_idx, epoch, x, y)
                else:
                    self._test_batch(epoch_metrics, batch_idx, epoch, test_batch)

class Solver():
    """The class is a wrapper around the entire training framework which
            most training parameters as input.

        Args:
            model: The model to be trained (VAE/CVAE/TDCVAE/TDCVAE2)
            data_loader: used for retrieving the data
            optimizer: The optimizing algorithm in use
            epochs: The number of epochs to train
            optim_config: config for the optimizer
            step_config: the config for the lr_scheduler
            lr_scheduler: learning scheduler for adaptive lr
            num_samples: the number of samples drawn each in epoch (and saved to file)
            cvae_mode: determines whether in cvae mode
            tdcvae_mode: determines whether in tdcvae mode
            tdcvae2_mode: determines whether in tdcvae2 mode
            save_model_state: determines whether model is saved after each epoch
    """
    def __init__(self, model, data_loader, optimizer, epochs, optim_config,\
            step_config=None, lr_scheduler=None, num_samples=100, cvae_mode=False,\
            tdcvae_mode=False, tdcvae2_mode=False, save_model_state=False):
        self.cvae_mode = cvae_mode
        self.tdcvae_mode = tdcvae_mode
        self.tdcvae2_mode = tdcvae2_mode
        self.data_loader = data_loader
        self.model = model
        self.model.to(DEVICE)
        self._set_weight_decay(optim_config)
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.epoch = 0
        self.epochs = epochs
        self.step_config = step_config
        self.lr_scheduler = lr_scheduler(self.optimizer, **step_config) if lr_scheduler else lr_scheduler
        self.train_loss_history = {x: [] for x in ["epochs", "train_loss_acc", "recon_loss_acc", "kl_diverg_acc"]}
        self.test_loss_history = []
        self.z_stats_history = {x: [] for x in ["mu_z", "std_z", "varmu_z", "expected_var_z"]}
        self.num_samples = num_samples

        if save_model_state and not self.data_loader.directories.make_dirs:
            raise ValueError("Can't save state if no folder is assigned to this run!")
        self.save_model_state = save_model_state

    def _set_weight_decay(self, optim_config):
        if optim_config["weight_decay"] is None:
            optim_config["weight_decay"] = 0.0
        elif optim_config["weight_decay"] == 1:
            # batch wise regularization, so M/N in all
            optim_config["weight_decay"] = 1/(float(self.data_loader.num_train_samples))

    def _save_train_metrics(self, epoch, metrics):
        num_train_samples = self.data_loader.num_train_samples
        num_train_batches = self.data_loader.num_train_batches
        # if rotation + scaling - we trigger MSE metric
        if self.tdcvae_mode and self.data_loader.thetas is not None and self.data_loader.scales is not None:
            train_loss = metrics.train_loss_acc/num_train_batches
            recon_loss = metrics.recon_loss_acc/num_train_batches
            kl_div = metrics.kl_diverg_acc/num_train_batches
        else:
            train_loss = metrics.train_loss_acc/num_train_samples
            recon_loss = metrics.recon_loss_acc/num_train_samples
            kl_div = metrics.kl_diverg_acc/num_train_samples
        self.train_loss_history["epochs"].append(epoch) # just for debug mode (in case we finish earlier)
        self.train_loss_history["train_loss_acc"].append(train_loss)
        self.train_loss_history["recon_loss_acc"].append(recon_loss)
        self.train_loss_history["kl_diverg_acc"].append(kl_div)
        self.z_stats_history["mu_z"].append(metrics.mu_z/num_train_batches)
        self.z_stats_history["std_z"].append(metrics.std_z/num_train_batches)
        self.z_stats_history["varmu_z"].append(metrics.varmu_z/num_train_batches)
        self.z_stats_history["expected_var_z"].append((metrics.expected_var_z/num_train_batches).item())
        return train_loss

    def _save_test_metrics(self, metrics):
        if self.tdcvae_mode and self.data_loader.thetas is not None and self.data_loader.scales is not None:
            test_loss = metrics.test_loss_acc/self.data_loader.num_test_batches
        else:
            test_loss = metrics.test_loss_acc/self.data_loader.num_test_samples
        self.test_loss_history.append(test_loss)
        return test_loss

    # generating samples from only the decoder
    def _sample(self, epoch):
        if not self.data_loader.directories.make_dirs:
            return
        with torch.no_grad():
            if self.cvae_mode:
                sample = cvae_sampling(self.model.decoder, self.num_samples, self.data_loader.n_classes, self.model.z_dim)
            elif self.tdcvae_mode:
                sample = tdcvae_sampling(self.model.decoder, self.data_loader.train_loader, self.data_loader.input_dim,\
                    self.num_samples, self.model.z_dim)
            elif self.tdcvae2_mode:
                x_decoded, x_t, x_next = tdcvae2_sampling(self.model.decoder, self.data_loader.train_loader,\
                    self.num_samples, self.model.z_dim)
                num_samples = min(self.num_samples, sample.size(0))
                recon_diff = x_decoded[:num_samples] - x_next[:num_samples]
                print("Sampling norm(sample, ground truth): {}".format(torch.norm(recon_diff)))
                recon_diff = recon_diff.pow(2)
                rd = recon_diff * 100
                recon_diff = x_next[:num_samples] - x_t[:num_samples]
                recon_diff = recon_diff.pow(2)
                rd2 = recon_diff * 100
                output = torch.cat([x_t[:num_samples],\
                                    x_next[:num_samples],
                                    x_decoded[:num_samples],
                                    rd,
                                    rd2])
                timestamps = self.data_loader.data.timestamps
                torchvision.utils.save_image(output,\
                    self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch) + "_t=["\
                    + ",".join(str(x) for x in timestamps[:num_samples]) + "]_z=" +\
                    str(self.model.z_dim) + ".png", nrow=self.num_samples)
                return
            else:
                sample = vae_sampling(self.model.decoder, self.num_samples, self.model.z_dim)
            num_samples = min(self.num_samples, sample.size(0))
            torchvision.utils.save_image(sample.view(num_samples, *self.data_loader.img_dims),\
                self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch)\
                + "_z=" + str(self.model.z_dim) + ".png", nrow=10)

    # save the model parameters to a txt file in the output folder
    def _save_model_params_to_file(self):
        if not self.data_loader.directories.make_dirs:
            return
        with open(self.data_loader.directories.result_dir + "/model_params_" +\
            self.data_loader.dataset + "_z=" + str(self.model.z_dim) + ".txt", 'w') as param_file:
            params = "epochs: {}\n"\
                "optimizer: {}\n"\
                "beta: {}\n"\
                "dim(z): {}\n"\
                "batch_size: {}\n"\
                "lr_scheduler: {}\n"\
                "step_config: {}\n"\
                .format(self.epochs, self.optimizer, self.model.beta, self.model.z_dim,\
                    self.data_loader.batch_size, self.lr_scheduler,\
                    self.step_config)
            params += "dataset: {}\n".format(self.data_loader.dataset)
            params += "VAE mode {}\n".format(not self.cvae_mode and not self.tdcvae_mode and not self.tdcvae2_mode)
            params += "CVAE mode: {}\n".format(self.cvae_mode)
            params += "TDCVAE mode: {}\n".format(self.tdcvae_mode)
            params += "TDCVAE2 mode: {}\n".format(self.tdcvae2_mode)
            if self.data_loader.thetas:
                params += "thetas: (theta_range_1: {}, theta_range_2: {})\n"\
                    .format(self.data_loader.theta_range_1, self.data_loader.theta_range_2)
            if self.data_loader.scales:
                params += "scales: (scale_range_1: {}, scale_range_2: {})\n"\
                    .format(self.data_loader.scale_range_1, self.data_loader.scale_range_2)
            params += "number of samples for sampling: {}\n".format(self.num_samples)
            if self.data_loader.resize:
                params += "resize: {}\n".format(self.data_loader.resize)
            if self.data_loader.crop:
                params += "crop: {}\n".format(self.data_loader.crop)
            params += "model:\n"
            params += str(self.model)
            param_file.write(params)
            print("params used:", params)

    # save the final model when training is done.
    def _save_final_model(self):
        name = self.data_loader.directories.result_dir + "/model_"
        if self.tdcvae_mode:
            name += "TDCVAE_"
            if self.data_loader.thetas and self.data_loader.scales:
                name += "SCALES_THETAS_"
            elif self.data_loader.thetas:
                name += "THETAS_"
            elif self.data_loader.scales:
                name += "SCALES_"
        elif self.cvae_mode:
            name += "CVAE_"
        elif self.tdcvae2_mode:
            name += "TDCVAE2_"
        else:
            name += "VAE_"
        last_train_loss = self.train_loss_history["train_loss_acc"][-1]
        name += self.data_loader.dataset + "_train_loss=" + "{0:.2f}".format(last_train_loss)\
            + "_z=" + str(self.model.z_dim) + ".pt"
        torch.save(self, name)

    def main(self):
        """The main method which training/testing of a model begins"""
        if self.data_loader.directories.make_dirs:
            print("+++++ START RUN | saved files in {} +++++".format(\
                self.data_loader.directories.result_dir_no_prefix))
        else:
            print("+++++ START RUN +++++ | no save mode")
        self._save_model_params_to_file()
        training = Training(self)
        testing = Testing(self)
        start = self.epoch if self.epoch else 1
        for epoch in range(start, self.epochs+1):
            epoch_watch = time.time()
            epoch_metrics = EpochMetrics()
            training.train(epoch_metrics)
            train_loss = self._save_train_metrics(epoch, epoch_metrics)
            print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss))
            testing.test(epoch, epoch_metrics)
            test_loss = self._save_test_metrics(epoch_metrics)
            print("====> Test set loss avg: {:.4f}".format(test_loss))
            self._sample(epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            if self.save_model_state:
                self.epoch = epoch+1 # signifying to continue from epoch+1 on.
                torch.save(self, self.data_loader.directories.result_dir + "/model_state.pt")
            print("{:.2f} seconds for epoch {}".format(time.time() - epoch_watch, epoch))
        if self.data_loader.directories.make_dirs:
            self._save_final_model()
        print("+++++ RUN IS FINISHED +++++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training a model (VAE/CVAE/TDCVAE/TDCVAE2)")
    parser.add_argument("--model", help="Set model to VAE/CVAE/TDCVAE/TDCVAE2 (required)", required=True)
    parser.add_argument("--dataset", help="Set dataset to MNIST/LFW/FF/LungScans accordingly (required, not case sensitive)", required=True)
    parser.add_argument("--save_files", help="Determine if files (samples etc.) should be saved (optional, default: False)", required=False, action='store_true')
    parser.add_argument("--save_model_state", help="Determine if state of model should be saved after each epoch during training (optional, default: False)",\
        required=False, action='store_true')
    parser.add_argument('--scales', help="Enables scaling of the model as specified in model_params", default=None, action='store_true')
    parser.add_argument('--thetas', help="Enables rotations of the model as specified in model_params", default=None, action='store_true')
    args = vars(parser.parse_args())
    model_arg = args["model"]
    dataset_arg = args["dataset"]
    save_files = args["save_files"]
    save_model_state = args["save_model_state"]

    if model_arg.lower() == "vae":
        data = get_model_data_vae(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg)
        model = Vae(data_loader.input_dim, data["hidden_dim"],\
            data["z_dim"], data["beta"], data["batch_norm"])
        solver = Solver(model, data_loader, data["optimizer"],\
            data["epochs"], data["optim_config"],\
            step_config=data["step_config"], lr_scheduler=data["lr_scheduler"],\
            save_model_state=save_model_state)
    elif model_arg.lower() == "cvae":
        data = get_model_data_cvae(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg)
        model = Cvae(data_loader.input_dim, data["hidden_dim"], data["z_dim"],\
            data["beta"], data_loader.n_classes, data["batch_norm"])
        solver = Solver(model, data_loader, data["optimizer"], data["epochs"],\
            data["optim_config"], step_config=data["step_config"],\
                lr_scheduler=data["lr_scheduler"], cvae_mode=True,\
                save_model_state=save_model_state)
    elif model_arg.lower() == "tdcvae":
        data = get_model_data_tdcvae(dataset_arg)
        scales = data["scales"] if args["scales"] is not None else None
        thetas = data["thetas"] if args["thetas"] is not None else None
        rotations = thetas is not None
        scaling = scales is not None
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg,\
            scales=scales, thetas=thetas)
        model = TD_Cvae(data_loader.input_dim,\
            data_loader.input_dim, data["z_dim"], data["beta"], rotations=rotations, scaling=scaling)
        solver = Solver(model, data_loader, data["optimizer"], data["epochs"],\
            data["optim_config"], step_config=data["step_config"],\
                lr_scheduler=data["lr_scheduler"], tdcvae_mode=True,\
                save_model_state=save_model_state)
    elif model_arg.lower() == "tdcvae2":
        # requires the data is at hand!
        root = "../data/lungscans"
        if not os.path.isdir(root):
            raise ValueError("Requires lung scan data is at {}".format(root))
        data = get_model_data_tdcvae2(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        folders = [[(root+"/"+f+"/"+a+"/") for a in os.listdir(root+"/"+f)] for f in os.listdir(root)]
        folders = [item for sublist in folders for item in sublist]
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg.lower(), resize=data["resize"], folders=folders)
        model = Tdcvae2(data["z_dim"], data["beta"], data["kernel_size_high"], data["kernel_size_low"])
        solver = Solver(model, data_loader, data["optimizer"], data["epochs"],\
            data["optim_config"], step_config=data["step_config"],\
                lr_scheduler=data["lr_scheduler"], tdcvae2_mode=True,\
                save_model_state=save_model_state, num_samples=8)
    solver.main()
