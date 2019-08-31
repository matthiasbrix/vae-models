"""This module contains all the parameters used for the different models
and their data sets in use. The purpose of having it centrally is to enable
training from the jupyter notebooks and the solver.py script, avoiding parameters
two different places. It gives also an overview of which data sets can be used
for each model.

"""
import torch
import numpy as np

def get_model_data_vae(dataset):
    """Parameters of the VAE model for the permitted datasets"""
    if dataset.lower() == "mnist":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 5,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 4,
            "batch_norm": False,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 300,
                "gamma" : 0.1
            },
            "optim_config": {
                "lr": 1e-3,
                "weight_decay": 1
            }
        }
    elif dataset.lower() == "lfw":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 30,
            "hidden_dim": 700,
            "z_dim": 20,
            "beta": 1,
            "batch_norm": True,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 30,
                "gamma" : 0.1
            },
            "optim_config": {
                "lr": 1e-2,
                "weight_decay": 1
            }
        }
    elif dataset.lower() == "ff":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 10,
            "hidden_dim": 200,
            "z_dim": 2,
            "beta": 1,
            "batch_norm": False,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 300,
                "gamma" : 0.1
            },
            "optim_config": {
                "lr": 1e-2,
                "weight_decay": 1
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params

def get_model_data_cvae(dataset):
    """Parameters of the VAE model for the permitted datasets"""
    if dataset.lower() == "mnist":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 3,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 1,
            "batch_norm": False,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 200,
                "gamma" : 0.1
            },
            "optim_config": {
                "lr": 1e-3,
                "weight_decay": 1
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params

def get_model_data_tdcvae(dataset):
    """Parameters of the TDCVAE model for the permitted datasets"""
    if dataset.lower() == "mnist":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 10000,
            "z_dim": 2,
            "beta": 0.00001,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 100,
                "gamma" : 0.75
            },
            "optim_config": {
                "lr": 1e-3,
                "weight_decay": None
            },
            "thetas": {
                "theta_1": None, #[-np.pi, np.pi],
                "theta_2": None #[-np.pi/4, np.pi/4]
            },
            "scales": {
                "scale_1": None, #[0.85, 1.15],
                "scale_2": None #[-0.15, 0.15]
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params

def get_model_data_tdcvae2(dataset):
    """Parameters of the TDCVAE2 model for the permitted datasets"""
    if dataset.lower() == "lungscans":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 32,
            "epochs": 1,
            "z_dim": 8,
            "beta": 1e-03,
            "resize": (64, 64),
            "kernel_size_high": 3,
            "kernel_size_low": 1,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 50,
                "gamma" : 0.75
            },
            "optim_config": {
                "lr": 3e-3,
                "weight_decay": None
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params
    