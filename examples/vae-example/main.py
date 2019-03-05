import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# TODO: Optim algorithm
# TODO: loss function

mnist_train = datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(type(mnist_train))
    print(type(mnist_test))
    print(train_loader)
    model = VAE().to(device)
