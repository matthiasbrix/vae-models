import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms

# TODO: Optim algorithm
# TODO: loss function

train_loader = torch.utils.data.DataLoader()

test_loader = torch.utils.data.DataLoader()

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self):
        pass

    def decode(self):
        pass

def train():
    pass

def test():
    pass

# call from here
if __name__ == "__main__":
    pass
