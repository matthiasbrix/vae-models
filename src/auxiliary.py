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
def produce_ys(encoder, x_t, scales, thetas, num_samples):
    ys = np.zeros((num_samples, scales.shape[0], thetas.shape[0], 2))
    encoder.eval()
    with torch.no_grad():
        for sample in range(ys.shape[0]):
            x_ts = np.reshape(np.tile(x_t[sample:sample+1, :], [thetas.shape[0], 1]), (thetas.shape[0], 28, 28))
            x_ts = torch.FloatTensor(np.expand_dims(x_ts, axis=1))
            for i in range(ys.shape[1]):
                #transform images and feed to the encoder, pick the mean opf y
                x_t_trans = preprocess_batch_det(x_ts, thetas, scales[i]*np.ones(thetas.shape[0]))
                x_t_trans = np.reshape(x_t_trans, (thetas.shape[0], 784))
                ys[sample, i, :, :] = encoder(torch.FloatTensor(x_t_trans))[0].detach().numpy()
    return ys

# transforming images to produce ys/zs
def produce_ys_zs(model, num_samples, x_t, scales_1, thetas_1, x_next=None, scales_2=None, thetas_2=None):
    if x_t.shape[0] is not x_next.shape[0]:
        return
    ys = np.zeros((num_samples, scales_1.shape[0], thetas_1.shape[0], 2))
    zs = np.zeros((num_samples, scales_2.shape[0], thetas_2.shape[0], 2))
    model.eval()
    with torch.no_grad():
        for sample in range(ys.shape[0]):
            x_ts = np.reshape(np.tile(x_t[sample:sample+1, :], [thetas_1.shape[0], 1]), (thetas_1.shape[0], 28, 28))
            x_ts = torch.FloatTensor(np.expand_dims(x_ts, axis=1))
            x_nexts = np.reshape(np.tile(x_next[sample:sample+1, :], [thetas_2.shape[0], 1]), (thetas_2.shape[0], 28, 28))
            x_nexts = torch.FloatTensor(np.expand_dims(x_nexts, axis=1))
            for i in range(ys.shape[1]):
                #transform images and feed to the encoder, pick the mean opf y
                x0trans = preprocess_batch_det(x_ts, thetas_1, scales_1[i]*np.ones(thetas_1.shape[0]))
                x0trans = np.reshape(x0trans, (thetas_1.shape[0], 784))
                x1trans = preprocess_batch_det(x_nexts, thetas_2, scales_2[i]*np.ones(thetas_2.shape[0]))
                x1trans = np.reshape(x1trans, (thetas_1.shape[0], 784))
                _, _, _, _, zs[sample, i, :, :], ys[sample, i, :, :] = model(torch.FloatTensor(x0trans), torch.FloatTensor(x1trans))
    return ys, zs
    