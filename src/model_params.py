import torch
import numpy as np

def get_model_data_vae(dataset):
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
                "weight_decay": None
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
                "weight_decay": None
            }
        }
    elif dataset.lower() == "ff":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 200,
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
                "weight_decay": None
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params

def get_model_data_cvae(dataset):
    if dataset.lower() == "mnist":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 10,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 1,
            "batch_norm": False,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 200,
                "gamma" : 0.1 # or 0.75
            },
            "optim_config": {
                "lr": 1e-3,
                "weight_decay": None
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params

def get_model_data_tdcvae(dataset):
    if dataset.lower() == "mnist":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 1000,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 2,
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
                "theta_1": [-np.pi, np.pi],
                "theta_2": [-np.pi/4, np.pi/4]
            },
            "scales": {
                "scale_1": [0.85, 1.15],
                "scale_2": [-0.15, 0.15]
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params
