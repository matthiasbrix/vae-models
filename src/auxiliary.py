import numpy as np
import torch
from preprocessing import preprocess_sample

import skimage as ski

# in case we have loaded a model to CPU but trained on GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# produces the z/y spaces for a single batch
def _get_batch_spaces(solver, x, y=None):
    y_space = None
    if solver.cvae_mode:
        x = x.view(-1, solver.data_loader.input_dim).to(DEVICE)
        y = y.to(DEVICE)
        _, _, _, z_space = solver.model(x, y)
    elif solver.tdcvae_mode:
        x_t, x_next = x
        x_t, x_next = x_t.view(-1, solver.data_loader.input_dim).to(DEVICE),\
        x_next.view(-1, solver.data_loader.input_dim).to(DEVICE)
        _, _, _, _, z_space, y_space = solver.model(x_t, x_next)
    else:
        x = x.view(-1, solver.data_loader.input_dim).to(DEVICE)
        _, _, _, z_space = solver.model(x) # vae
    return y_space, z_space

# Used for retrieving the latent spaces available (z and y)
# transformation is only set when using rotation and/or scaling.
def get_latent_spaces(solver, transformation=None):
    z_space = np.zeros((solver.data_loader.num_test_samples, solver.model.z_dim))
    y_space = np.zeros((solver.data_loader.num_test_samples, solver.model.z_dim))
    data_labels = np.zeros((solver.data_loader.num_test_samples))
    test_loader = solver.data_loader.get_new_test_data_loader()
    solver.model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            batch_start_idx = batch_idx*solver.data_loader.batch_size
            batch_end_idx = (batch_idx+1)*solver.data_loader.batch_size
            if solver.data_loader.with_labels:
                x, targets = data[0], data[1]
                if transformation is not None:
                    x = transformation.preprocess_batch(x, batch_start_idx, batch_end_idx)
                y_batch_space, z_batch_space = _get_batch_spaces(solver, x, targets)
                data_labels[batch_start_idx:batch_end_idx] = targets
            else:
                x = data
                if transformation is not None:
                    x = transformation.preprocess_batch(x, batch_start_idx, batch_end_idx)
                y_batch_space, z_batch_space = _get_batch_spaces(solver, x)
            z_space[batch_start_idx:batch_end_idx, :] = z_batch_space.cpu().numpy()
            if y_batch_space is not None:
                y_space[batch_start_idx:batch_end_idx, :] = y_batch_space.cpu().numpy()
    return z_space, y_space, data_labels

# transforming images to produce alphas/radiuses
def transform_images(solver, preprocessing, test_loader, ys):
    solver.model.eval()
    with torch.no_grad():
        data_labels = np.zeros((solver.data_loader.num_test_samples))
        for _, data in enumerate(test_loader):
            if solver.data_loader.with_labels:
                x_t, targets = data[0], data[1]
            else:
                x_t = data
            num_samples = ys.shape[2]
            data_labels[:num_samples] = targets[:num_samples]
            sample_idx = 0
            # do transformation on each sample for each scale and then over all thetas
            for sample_idx in range(num_samples):
                for s in range(ys.shape[0]):
                    for t in range(ys.shape[1]):
                        scale = np.around(preprocessing.scales[s], decimals=2)
                        theta = np.around(preprocessing.thetas[t], decimals=2)
                        print(scale, theta)
                        x_transformed = preprocess_sample(x_t[sample_idx], theta=theta, scale=(scale, scale)).view(-1, solver.data_loader.input_dim).to(DEVICE)
                        _, _, _, _, _, y_batch_space = solver.model(x_transformed, None)
                        print(y_batch_space.shape)
                        ys[s, t, sample_idx, :] = y_batch_space[0].cpu().numpy()
            return



#data preprocessing for rotation learning
def transform_batch(x, theta, s):
    batch_size = x.shape[0]
    xtransformed=[]
    for i in range(batch_size):
        shift_y, shift_x = np.array(x.shape[1:3]) / 2.
        center_shift = ski.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        center_shift_inv = ski.transform.SimilarityTransform(translation=[shift_x, shift_y])
        center_transform = ski.transform.AffineTransform(scale=(s[i], s[i]), rotation=np.radians(theta[i]))
        transformation = center_shift + (center_transform + center_shift_inv)
        xtransformed.append(ski.transform.warp(x[i,:,:], transformation.inverse, output_shape=(x.shape[1], x.shape[2]), preserve_range=True))
    x=np.stack(xtransformed,axis=0)
    return x

def transform_images2(solver, preprocessing, test_loader, ys, theta, s):
    solver.model.eval()
    with torch.no_grad():
        x_t, _ = iter(test_loader).next()
        print(x_t.shape)
        # do transformation on each sample for each scale and then over all thetas
        for sample in range(ys.shape[0]):
            print("sample", sample)
            #create one copy of the sample for each theta we want to use
            x0tile = np.reshape(np.tile(x_t[sample:sample+1, :], [theta.shape[0], 1]), (theta.shape[0], 28, 28))
            for i in range(ys.shape[1]):
                #transform images and feed to the encoder, pick the mean opf y
                x0trans= transform_batch(x0tile, theta, s[i]*np.ones(theta.shape[0]))
                x0trans= torch.FloatTensor(np.reshape(x0trans,(theta.shape[0], 784)))
                _, _, _, _, _, y_batch_space = solver.model(x0trans) # outputs 30, 784; ys[sample,i,:,:].shape is 30, 2
                ys[sample,i,:,:] = y_batch_space.numpy()
