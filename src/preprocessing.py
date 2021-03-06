"""This module contains preprocessing procedures for the TDCVAE model"""
from random import uniform
import numpy as np
import torch
import skimage as ski

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_angles(theta_range_1, theta_range_2, bound_angle=None):
    """Generating angles uniformly random in radians given two ranges. If one of them is None
    we trigger the built-in default values.
        Args:
            theta_range_1: A range of angles in radians for theta_1
            theta_range_2: A range of angles in radians for theta_2
            bound_angle: The min/max angle bound for theta_2 if using built-in ranges
    """
    if theta_range_1 is None or theta_range_2 is None:
        theta_1 = -np.pi + np.random.uniform() * 2 * np.pi
        bound_angle = np.pi/4 if bound_angle is None else bound_angle
        theta_2 = theta_1 - bound_angle + np.random.uniform() * 2 * bound_angle
    else:
        theta_1 = uniform(*theta_range_1)
        theta_2 = theta_1 + uniform(*theta_range_2)
    theta_1 = np.around(theta_1, decimals=2)
    theta_2 = np.around(theta_2, decimals=2)
    return theta_1, theta_2

def generate_scales(scale_range_1, scale_range_2):
    """Generating scales uniformly random given two ranges. If one of them is None
    we trigger the built-in default values.
        Args:
            scale_range_1: A range of scales for scale_1
            scale_range_2: A range of scales for scale_2
    """
    if scale_range_1 is None or scale_range_2 is None:
        scale_1 = 0.85 + np.random.uniform() * 0.3
        scale_2 = scale_1 - 0.15 + np.random.uniform() * 0.3
    else:
        scale_1 = uniform(*scale_range_1)
        scale_2 = scale_1 + uniform(*scale_range_2)
    scale_1 = np.around(scale_1, decimals=2)
    scale_2 = np.around(scale_2, decimals=2)
    return scale_1, scale_2

def preprocess_sample(x, theta=None, scale=None):
    """Applying affine transformation to a given tensor from its shifted center (0,0)
        Args:
            x: A tensor/image (1, H, W)
            theta: An angle in radian (default: None, i.e. 0)
            scale: A scale (default: None, i.e. 1.0)
    """
    x = x[0].numpy()
    theta = 0 if theta is None else theta
    scale = (1.0, 1.0) if scale is None or not all(scale) else scale
    shift_y, shift_x = np.array(x.shape[-2:])/2.
    center_shift = ski.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    center_shift_inv = ski.transform.SimilarityTransform(translation=[shift_x, shift_y])
    center_transform = ski.transform.AffineTransform(scale=scale, rotation=theta)
    transformation = center_shift + (center_transform + center_shift_inv)
    # apply transformation inversed because otherwise it would be opposite/inverse of what we define, so 1.5 scale is actually 0.5
    return torch.FloatTensor(ski.transform.warp(x, transformation.inverse, output_shape=(x.shape[0], x.shape[1]), preserve_range=True)).to(DEVICE)

def preprocess_batch_det(x, thetas, scales):
    """Preprocessing a batch with given thetas,scales
        Args:
            x: numpy array (N, 1, H, W)
            thetas: numpy array / list of N angles in radians
            scales: numpy array / list of N scales
    """
    batch_size = x.shape[0]
    preprocessed_x = []
    for i in range(batch_size):
        preprocessed_x.append(preprocess_sample(x[i], theta=thetas[i], scale=(scales[i], scales[i])))
    preprocessed_x = np.stack(preprocessed_x, axis=0)
    return preprocessed_x

class RandomPreprocessing():
    """This class is for randomly applying transformation on a whole data set (e.g. test set)
    and saves the parameters for labeling of plot. We use this especially for producing
    the latent spaces.
        Args:
            num_test_samples: The number of test samples
            img_dims: The image dimensions of a single example
            theta_range_1: The range of angles in radians for theta_1
            theta_range_2: The range of angles in radians for theta_2
            scale_range_1: The range of scales for scale_1
            scale_range_2: The range of scales for scale_2
    """
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

    def _save_params(self, start_idx, end_idx):
        if self.rotations:
            theta_diff = np.around(self.theta_2 - self.theta_1, decimals=2)
            self.prepro_params["theta_diff"][start_idx:end_idx] = theta_diff
            self.prepro_params["theta_1"][start_idx:end_idx] = self.theta_1
        if self.scaling:
            scale_diff = np.around(self.scale_2 - self.scale_1, decimals=2)
            self.prepro_params["scale_diff"][start_idx:end_idx] = scale_diff
            self.prepro_params["scale_1"][start_idx:end_idx] = self.scale_1

    def preprocess_samples(self, x, batch_idx=None, save=True):
        """Preprocesses a batch given the assigned ranges in the class and saves the parameters
            Args:
                x: The batch to preprocess.
                batch_idx: The batch index of the data set.
                save: Determines whether to save the parameters or not.
        """
        x_t = torch.zeros_like(x)
        x_next = torch.zeros_like(x)
        xt, xt1 = x.clone().detach(), x.clone().detach()
        for i in range(x_t.shape[0]):
            if self.rotations:
                self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
            else:
                self.theta_1, self.theta_2 = 0, 0
            if self.scaling:
                self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
            else:
                self.scale_1, self.scale_2 = 1.0, 1.0
            x_t[i] = preprocess_sample(xt[i], self.theta_1, (self.scale_1, self.scale_1))
            x_next[i] = preprocess_sample(xt1[i], self.theta_2, (self.scale_2, self.scale_2))
            if save:
                self._save_params(batch_idx+i, batch_idx+i+1)
        return x_t, x_next
