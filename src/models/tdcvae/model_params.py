import torch

def get_data_model(dataset):
    if dataset == "MNIST":
        return {
            "optimizer": torch.optim.Adam,
            "batch_size": 128,
            "epochs": 5,
            "hidden_dim": 500,
            "z_dim": 2,
            "beta": 1,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 100,
                "gamma" : 0.75 # or 0.1
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
    else:
        raise ValueError("Dataset not known!")
