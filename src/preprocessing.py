import numpy as np
import torch
from random import uniform
import skimage as ski

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_angles(theta_range_1, theta_range_2, bound_rad=None):
    if theta_range_1 is None and theta_range_2 is None:
        theta_1 = -np.pi + np.random.uniform() * 2 * np.pi
        bound_rad = np.pi/4 if bound_rad is None else bound_rad
        theta_2 = theta_1 -bound_rad + np.random.uniform() * 2 * bound_rad
    else:
        theta_1 = uniform(*theta_range_1)
        theta_2 = theta_1 + uniform(*theta_range_2)
    theta_1 = np.around(theta_1, decimals=2)
    theta_2 = np.around(theta_2, decimals=2)
    return theta_1, theta_2

def generate_scales(scale_range_1, scale_range_2):
    if scale_range_1 is None and scale_range_2 is None:
        scale_1 = 0.85 + np.random.uniform() * 0.3
        scale_2 = scale_1 - 0.15 + np.random.uniform() * 0.3
    else:
        scale_1 = np.around(uniform(*scale_range_1), decimals=2)
        scale_2 = scale_1 + np.around(uniform(*scale_range_2), decimals=2)
    scale_1 = np.around(scale_1, decimals=2)
    scale_2 = np.around(scale_2, decimals=2)
    return scale_1, scale_2

# expecting a tensor (1, H, W) so not working on multuple channels.
def preprocess_sample(x, theta=None, scale=None):
    x = x[0].numpy()
    theta = 0 if theta is None else theta
    scale = (1.0, 1.0) if scale is None or not all(scale) else scale
    shift_y, shift_x = np.array(x.shape[-2:])/2.
    center_shift = ski.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    center_shift_inv = ski.transform.SimilarityTransform(translation=[shift_x, shift_y])
    center_transform = ski.transform.AffineTransform(scale=scale, rotation=theta)
    transformation = center_shift + (center_transform + center_shift_inv)
    # apply transformation inversed because otherwise it would be opposite/inverse of we define, so 1.5 scale is actually 0.5
    return torch.FloatTensor(ski.transform.warp(x, transformation.inverse, output_shape=(x.shape[0], x.shape[1]), preserve_range=True)).to(DEVICE)

# expecting numpy array (N, H, W)
def preprocess_batch_det(x, thetas, scales):
    batch_size = x.shape[0]
    preprocessed_x = []
    for i in range(batch_size):
        preprocessed_x.append(preprocess_sample(x[i], theta=thetas[i], scale=(scales[i], scales[i])))
    preprocessed_x = np.stack(preprocessed_x, axis=0)
    return preprocessed_x

# This is for the latent spaces which just need random transformations and want to save the prepro params
class RandomPreprocessing():
    def __init__(self, num_test_samples, img_dims, theta_range_1=None, theta_range_2=None, scale_range_1=None, scale_range_2=None):
        self.prepro_params = {}
        self.rotations = theta_range_1 is not None and theta_range_2 is not None
        self.scaling = scale_range_1 is not None and scale_range_2 is not None
        self.img_dims = img_dims
        if self.rotations:
            self.theta_range_1 = theta_range_1
            self.theta_range_2 = theta_range_2
            self.theta_1, self.theta_2 = 0.0, 0.0
            self.prepro_params["theta_1"] = np.zeros((num_test_samples))
            self.prepro_params["theta_diff"] = np.zeros((num_test_samples))
        if self.scaling:
            self.scale_range_1 = scale_range_1
            self.scale_range_2 = scale_range_2
            self.scale_1, self.scale_2 = 0.0, 0.0
            self.prepro_params["scale_1"] = np.zeros((num_test_samples))
            self.prepro_params["scale_diff"] = np.zeros((num_test_samples))

    def _save_params(self, batch_start_idx, batch_end_idx):
        if self.rotations:
            theta_diff = np.around(self.theta_2 - self.theta_1, decimals=2)
            self.prepro_params["theta_diff"][batch_start_idx:batch_end_idx] = theta_diff
            self.prepro_params["theta_1"][batch_start_idx:batch_end_idx] = self.theta_1
        if self.scaling:
            scale_diff = np.around(self.scale_2 - self.scale_1, decimals=2)
            self.prepro_params["scale_diff"][batch_start_idx:batch_end_idx] = scale_diff
            self.prepro_params["scale_1"][batch_start_idx:batch_end_idx] = self.scale_1

    def preprocess_samples(self, x, batch_idx=None, save=True):
        x_t = torch.zeros_like(x)
        x_next = torch.zeros_like(x)
        x0, x1 = x.clone().detach(), x.clone().detach()
        print(self.rotations, self.scaling)
        for i in range(x_t.shape[0]):
            if self.rotations:
                self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
            else:
                self.theta_1, self.theta_2 = 0, 0
            if self.scaling:
                self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
            else:
                self.scale_1, self.scale_2 = 1.0, 1.0
            x_t[i] = preprocess_sample(x0[i], self.theta_1, (self.scale_1, self.scale_1))
            x_next[i] = preprocess_sample(x1[i], self.theta_2, (self.scale_2, self.scale_2))
            if save:
                self._save_params(batch_idx+i, batch_idx+i+1)
        return x_t, x_next