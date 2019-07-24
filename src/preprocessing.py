import numpy as np
import torch
from random import uniform
import skimage as ski

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_angles(theta_range_1, theta_range_2):
    theta_1 = -np.pi + np.random.uniform() * 2 * np.pi # np.around(uniform(*theta_range_1), decimals=2)
    theta_2 = theta_1 -np.pi/4 + np.random.uniform() * 2* np.pi/4 # theta_1 + np.around(uniform(*theta_range_2), decimals=2)
    return theta_1, theta_2

def generate_scales(scale_range_1, scale_range_2):
    scale_1 = 0.85 + np.random.uniform() * 0.3 # np.around(uniform(*scale_range_1), decimals=2)
    scale_2 = scale_1-0.15 + np.random.uniform() * 0.3 # scale_1 + np.around(uniform(*scale_range_2), decimals=2)
    return scale_1, scale_2

# expecting a tensor (C, H, W)
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
    return torch.FloatTensor(ski.transform.warp(x, transformation.inverse, output_shape=(x.shape[-2], x.shape[-1]), preserve_range=True)).to(DEVICE)

# This is for the latent spaces which just need random transformations and want to save the prepro params
class RandomPreprocessing():
    def __init__(self, num_test_samples, img_dims, theta_range_1=None, theta_range_2=None, scale_range_1=None, scale_range_2=None):
        self.prepro_params = {}
        self.rotations = theta_range_1 is not None and theta_range_2 is not None
        self.scaling = scale_range_1 is not None and scale_range_2 is not None
        self.img_dims = img_dims
        # TODO: Don't have theta_1 and scale_1 as class props
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

    def _apply_transformation(self, x, theta=None, scale=None):
        x_transformed = torch.zeros_like(x)
        for i in range(x.shape[0]):
            if self.rotations and self.scaling:
                x_transformed[i] = preprocess_sample(x[i], theta=theta, scale=scale)
            elif self.scaling:
                x_transformed[i] = preprocess_sample(x[i], scale=scale)
            elif self.rotations:
                x_transformed[i] = preprocess_sample(x[i], theta=theta)
        return x_transformed

    def preprocess_batch(self, x, batch_start_idx=None, batch_end_idx=None):
        if self.rotations and self.scaling:
            self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
            self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
            x_t = self._apply_transformation(x, theta=self.theta_1, scale=(self.scale_1, self.scale_1))
            x_next = self._apply_transformation(x, theta=self.theta_2, scale=(self.scale_2, self.scale_2))
        elif self.scaling:
            self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
            x_t = self._apply_transformation(x, scale=(self.scale_1, self.scale_1))
            x_next = self._apply_transformation(x, scale=(self.scale_2, self.scale_2))
        elif self.rotations:
            self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
            x_t = self._apply_transformation(x, theta=self.theta_1)
            x_next = self._apply_transformation(x, theta=self.theta_2)
        else:
            raise ValueError("Prepro of batch failed")
        if batch_start_idx is not None and batch_end_idx is not None:
            self._save_params(batch_start_idx, batch_end_idx)
        return x_t, x_next

# For producing y spaces with deterministic deltas (scales/thetas)
class DeterministicPreprocessing():
    def __init__(self, num_test_samples, img_dims, num_rotations, num_scales):
        self.num_test_samples = num_test_samples
        self.img_dims = img_dims
        if num_scales <= 0 or num_rotations <= 0:
            raise ValueError("Det. prepro failed because rotations and/or scales should be > 0")
        if num_rotations > 0:
            self.thetas = np.linspace(-np.pi, np.pi, num_rotations)
        if num_scales > 0:
            self.scales = 0.85 + np.linspace(0, 1, num_scales) * 0.3

    '''
    def preprocess_batch(self, x, scale=None, theta=None):
        x_transformed = torch.zeros_like(x)
        for i in range(x.shape[0]):
            if scale is not None and theta is not None:
                x_transformed[i] = preprocess_sample(x[i], theta=theta, scale=(scale, scale))
            elif scale is not None:
                x_transformed[i] = preprocess_sample(x[i], scale=(scale, scale))
            elif theta is not None:
                x_transformed[i] = preprocess_sample(x[i], theta=theta)
        return x_transformed
    '''