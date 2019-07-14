import torch

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
            "epochs": 100,
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
            "epochs": 5,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 1,
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
                "theta_1": [0, 360],
                "theta_2": [0, 60]
            },
            "scales": {
                "scale_1": [0.7, 1.3],
                "scale_2": [0.2, 0.5]
            }
        }
    elif dataset.lower() == "lungscans":
        params = {
            "optimizer": torch.optim.Adam,
            "batch_size": 4,
            "epochs": 1,
            "hidden_dim": 1000,
            "z_dim": 2,
            "beta": 1,
            "resize": (80, 80),
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 100,
                "gamma" : 0.75
            },
            "optim_config": {
                "lr": 1e-4,
                "weight_decay": None
            },
            "thetas": {
                "theta_1": [-45, 45],
                "theta_2": [-10, 10]
            },
            "scales": {
                "scale_1": [1.0, 1.3],
                "scale_2": [0.2, 0.5]
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params
