import numpy as np
import torch

from preprocessing import preprocess_batch_det

# in case we have loaded a model to CPU but trained on GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# produces the z/y spaces for a single batch
def _get_batch_spaces(model, input_dim, x, mode, y=None):
    y_space = None
    if mode == "cvae":
        x = x.view(-1, input_dim).to(DEVICE)
        y = y.to(DEVICE)
        _, _, _, z_space = model(x, y)
    elif mode == "tdcvae":
        x_t, x_next = x
        x_t, x_next = x_t.view(-1, input_dim).to(DEVICE),\
        x_next.view(-1, input_dim).to(DEVICE)
        _, _, _, _, z_space, y_space = model(x_t, x_next)
    else:
        x = x.view(-1, input_dim).to(DEVICE)
        _, _, _, z_space = model(x) # vae
    return y_space, z_space

# Used for retrieving the latent spaces available (z and y)
# transformation is only set when using rotation and/or scaling.
def get_latent_spaces(model, mode, test_loader, num_test_samples, z_dim, batch_size,\
    with_labels, input_dim, transformation=None):
    z_space = np.zeros((num_test_samples, z_dim))
    y_space = np.zeros((num_test_samples, z_dim))
    data_labels = np.zeros((num_test_samples))
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            batch_start_idx = batch_idx*batch_size
            batch_end_idx = (batch_idx+1)*batch_size
            if with_labels:
                x, targets = data[0], data[1]
                if transformation is not None:
                    x = transformation.preprocess_samples(x, batch_start_idx)
                y_batch_space, z_batch_space = _get_batch_spaces(model, input_dim, x, mode, targets)
                data_labels[batch_start_idx:batch_end_idx] = targets
            else:
                x = data
                if transformation is not None:
                    x = transformation.preprocess_samples(x, batch_start_idx)
                y_batch_space, z_batch_space = _get_batch_spaces(model, input_dim, x, mode)
            z_space[batch_start_idx:batch_end_idx, :] = z_batch_space.cpu().numpy()
            if y_batch_space is not None:
                y_space[batch_start_idx:batch_end_idx, :] = y_batch_space.cpu().numpy()
    return z_space, y_space, data_labels

# transforming images to produce alphas/radiuses
def produce_alphas_radiuses(encoder, x_t, scales, thetas, num_samples, num_scales, num_rotations):
    ys = np.zeros((num_samples, num_scales, num_rotations, 2))
    encoder.eval()
    with torch.no_grad():
        for sample in range(ys.shape[0]):
            x0tile = np.reshape(np.tile(x_t[sample:sample+1, :],[thetas.shape[0], 1]), (thetas.shape[0], 28, 28))
            x0tile = torch.FloatTensor(np.expand_dims(x0tile, axis=1))
            for i in range(ys.shape[1]):
                #transform images and feed to the encoder, pick the mean opf y
                x0trans = preprocess_batch_det(x0tile, thetas, scales[i]*np.ones(thetas.shape[0]))
                x0trans = np.reshape(x0trans, (thetas.shape[0], 784))
                ys[sample, i, :, :] = encoder(torch.FloatTensor(x0trans))[0].detach().numpy()
    return ys
