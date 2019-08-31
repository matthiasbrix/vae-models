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

# 
def get_latent_spaces(model, mode, test_loader, num_test_samples, z_dim, batch_size,\
    with_labels, input_dim, transformation=None):
    """Computes the latent spaces available (z and y) on the test data set
    
        Note:
            transformation is only set when using rotation and/or scaling (for tdcvae)
    """
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

def produce_ys(encoder, x_t, scales, thetas, num_samples):
    """Transforming images, encoding them to produce y space (for alpha/radiuses)"""
    ys = np.zeros((num_samples, scales.shape[0], thetas.shape[0], 2))
    encoder.eval()
    with torch.no_grad():
        for sample in range(ys.shape[0]):
            x_ts = np.reshape(np.tile(x_t[sample:sample+1, :], [thetas.shape[0], 1]), (thetas.shape[0], 28, 28))
            x_ts = torch.FloatTensor(np.expand_dims(x_ts, axis=1))
            for i in range(ys.shape[1]):
                #transform images and feed to the encoder, pick the mean opf y
                xt_trans = preprocess_batch_det(x_ts, thetas, scales[i]*np.ones(thetas.shape[0]))
                xt_trans = np.reshape(xt_trans, (thetas.shape[0], 784))
                ys[sample, i, :, :] = encoder(torch.FloatTensor(xt_trans))[0].detach().numpy()
    return ys

def produce_ys_zs(model, num_samples, x_t, scales_1, thetas_1, x_next, scales_2, thetas_2):
    """Transforming images, encoding them to produce y/z spaces"""
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
                xt_trans = preprocess_batch_det(x_ts, thetas_1, scales_1[i]*np.ones(thetas_1.shape[0]))
                xt_trans = np.reshape(xt_trans, (thetas_1.shape[0], 784))
                xnext_trans = preprocess_batch_det(x_nexts, thetas_2, scales_2[i]*np.ones(thetas_2.shape[0]))
                xnext_trans = np.reshape(xnext_trans, (thetas_1.shape[0], 784))
                _, _, _, _, zs[sample, i, :, :], ys[sample, i, :, :] = model(torch.FloatTensor(xt_trans), torch.FloatTensor(xnext_trans))
    return ys, zs
    
# copied from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
def remap_range(x, old_min, old_max, n_min, n_max):
    """mapping numbers of a range [old_min,old_max] to [n_min, n_max]"""
    # range check
    if old_min == old_max:
        print("Warning: Zero input range")
        return None
    if n_min == n_max:
        print("Warning: Zero output range")
        return None
    # check reversed input range
    reverse_input = False
    old_min = min(old_min, old_max)
    old_max = max(old_min, old_max)
    if not old_min == old_min:
        reverse_input = True
    # check reversed output range
    reverse_output = False
    new_min = min(n_min, n_max)
    new_max = max(n_min, n_max)
    if not new_min == n_min:
        reverse_output = True
    portion = (x-old_min)*(new_max-new_min)/(old_max-old_min)
    if reverse_input:
        portion = (old_max-x)*(new_max-new_min)/(old_max-old_min)
    result = portion + new_min
    if reverse_output:
        result = new_max - portion
    return result