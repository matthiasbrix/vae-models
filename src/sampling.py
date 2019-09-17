import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def vae_sampling(decoder, num_samples, z_dim):
    with torch.no_grad():
        sample = torch.randn((num_samples, z_dim))
        decoded = decoder(sample)
        return decoded

def cvae_sampling(decoder, num_samples, n_classes, z_dim):
    with torch.no_grad():
        z_sample = torch.randn(num_samples, z_dim)
        idx = torch.randint(0, n_classes, (1,)).item()
        y_sample = torch.FloatTensor(torch.zeros(z_sample.size(0), n_classes))
        y_sample[:, idx] = 1.
        sample = torch.cat((z_sample, y_sample), dim=-1).to(DEVICE)
        decoded = decoder(sample)
        return decoded

def tdcvae_sampling(decoder, test_set_loader, input_dim, num_samples, z_dim):
    with torch.no_grad():
        x_t = iter(test_set_loader).next()[0][0]
        x_t = x_t.view(-1, input_dim)
        num_samples = min(num_samples, x_t.size(0))
        x_t = x_t[:num_samples]
        z_sample = torch.randn(x_t.size(0), z_dim).to(DEVICE)
        sample = torch.cat((x_t, z_sample), dim=-1).to(DEVICE)
        decoded = decoder(sample)
        return decoded
    
def tdcvae2_sampling(decoder, test_set_loader, num_samples, z_dim):
    with torch.no_grad():
        x_t, x_next = iter(test_set_loader).next()
        num_samples = min(num_samples, x_t.size(0))
        x_t = x_t[:num_samples]
        z_sample = torch.randn(x_t.shape[0], z_dim, x_t.shape[2], x_t.shape[3])
        sample = torch.cat((x_t, z_sample), dim=1).to(DEVICE)
        decoded = decoder(sample)
        return decoded, x_t, x_next