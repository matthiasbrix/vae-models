import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST(root="../data/MNIST", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="../data/MNIST", train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=1, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, Din, H, Dout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear21 = nn.Linear(H, Dout) # \mu(X)
        self.linear22 = nn.Linear(H, Dout) # \sum(X)

    # compute Q(z|X) whic his encoding X into z
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear21(x)), F.relu(self.linear22(x)) # \mu(X), \sum(X) so mean(X) and covariance(X)

class Decoder(nn.Module):
    def __init__(self, Dout, H, Din):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(Dout, H)
        self.linear2 = nn.Linear(H, Din)
        self.sigmoid = nn.Sigmoid()

    # compute P(X|z) (posterior) which is decoding to reconstruct X
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.sigmoid(self.linear2(x))

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # sampling from N(\mu(X), \sum(X))
    def reparameterization_trick(self, mu, logsigma):
        sigma = torch.exp(1/2*logsigma) # TODO why 1/2?
        eps = torch.randn_like(sigma) # sampling eps ~ N(0, I)
        return mu + sigma*eps # compute z = \mu(X) + \sum^{1/2}(X) * eps

    def forward(self, data):
        mu, logsigma = self.encoder(data)
        z = self.reparameterization_trick(mu, logsigma)
        decoded = self.decoder(z)
        return decoded, mu, logsigma

# loss function + KL divergence, use for this \mu(X), \sum(X)
# compute here D_{KL}[N(\mu(X), \sum(X))||N(0,1)] = 1/2 \sum_k (\sum(X)+\mu^2(X) - 1 - log \sum(X))
def loss_function(fx, X, logsigma, mu):
    bce = F.binary_cross_entropy(fx, X)
    kl_divergence = 1/2 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) # by appendix B in the Auto Encoding Variational Bayes
    #kl_divergence2 = 1/2 * torch.sum(logsigma.exp() + mu.pow(2) - 1 - logsigma) # will give same value but negative would need to + below
    return bce - kl_divergence

def train(vae_model, optimizer, epoch):
    vae_model = vae_model.train()
    train_loss = 0
    for _, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        X = data.view(-1, 28 * 28)
        decoded, mu, logsigma = vae_model(X) # shapes are torch.Size([100, 784]) torch.Size([100, 20]) torch.Size([100, 20]) because batch_size = 100
        loss = loss_function(decoded, X, logsigma, mu)
        loss.backward() # compute gradients
        train_loss += loss.item()
        optimizer.step()
    print("====> Epoch: {} loss: {:.4f}".format(
          epoch, train_loss))
    return train_loss

# copied from https://github.com/pytorch/examples/blob/master/vae/main.py
def test(vae_model, epoch):
    vae_model = vae_model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.view(-1, 28 * 28).to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            #if i == 0:
            #    n = min(data.size(0), 8)
            #    comparison = torch.cat([data[:n], recon_batch.view(64, 1, 28, 28)[:n]])
            #    torchvision.utils.save_image(comparison.cpu(), "testing/test_reconstruction_" + str(epoch) + ".png", nrow=n)
    print("====> Test set loss: {:.4f}".format(test_loss / len(test_loader.dataset)))

if __name__ == "__main__":
    os.makedirs("testing", exist_ok=True)
    hidden_units = 500 # Kingma, Welling use 500 neurons, otherwise use 400
    z_dim = 20 # 1000 is suggested in the paper "Tutorial on VAE" but Kingma, Welling show 20 is sufficient for MNIST
    encoder = Encoder(784, hidden_units, z_dim)
    decoder = Decoder(z_dim, hidden_units, 784)
    model = VAE(encoder, decoder)
    optimalg = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999)) # params is iterable of parameters to optimize or dicts defining parameter groups
    model = model.to(device)
    epochs = 10000
    scheduler = optim.lr_scheduler.StepLR(optimalg, step_size=20, gamma=0.1)
    prev_train_loss = train(model, optimalg, 1)
    test(model, 1)
    for epoch in range(2, epochs+1):
        scheduler.step()
        train_loss = train(model, optimalg, epoch)
        test(model, epoch)
        with torch.no_grad():
            # In test time we disregard the encoder and only generate z from N(0,I) which we use as arg to decoder
            sample = torch.randn(64, z_dim).to(device)
            sample = model.decoder(sample).cpu()
            torchvision.utils.save_image(sample.view(64, 1, 28, 28), "testing/test_sample_" + str(epoch) + "_z=" + str(z_dim) + ".png") # inserting a mini batch tensor to compute a grid
        if abs(train_loss - prev_train_loss) < 1e-7:
            break
        prev_train_loss = train_loss

# digits 3,5 and 8,0 look similar
# TODO visualize the latent space like here: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/