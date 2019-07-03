import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import uniform

# This is for the latent spaces which just need random transformations
class RandomPreprocessing():
    def __init__(self, num_test_samples, img_dims, theta_range_1=None, theta_range_2=None, scale_range_1=None, scale_range_2=None):
        self.prepro_params = {}
        self.rotations = theta_range_1 is not None and theta_range_2 is not None
        self.scaling = scale_range_1 is not None and scale_range_2 is not None
        self.img_dims = img_dims
        if self.rotations:
            self.theta_range_1 = theta_range_1
            self.theta_range_2 = theta_range_2
            self.theta_1, self.theta_2 = 0, 0
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
            theta_diff = self.theta_2 - self.theta_1
            self.prepro_params["theta_diff"][batch_start_idx:batch_end_idx] = theta_diff
            self.prepro_params["theta_1"][batch_start_idx:batch_end_idx] = self.theta_1
        if self.scaling:
            scale_diff = self.scale_2 - self.scale_1
            self.prepro_params["scale_diff"][batch_start_idx:batch_end_idx] = scale_diff
            self.prepro_params["scale_1"][batch_start_idx:batch_end_idx] = self.scale_1

    def _generate_scales(self):
        self.scale_1 = np.around(uniform(*self.scale_range_1), decimals=2)
        self.scale_2 = np.around(self.scale_1 + uniform(*self.scale_range_2), decimals=2)
        if self.scale_1 <= 0 or self.scale_2 <= 0:
            raise ValueError("One of the scales is <= 0!")

    def _generate_angles(self):
        self.theta_1 = np.random.randint(*self.theta_range_1)
        self.theta_2 = self.theta_1 + np.random.randint(*self.theta_range_2)

    # check if how much scaled shape will be <= self.data_loader.img_dims and then pad accordingliy
    def _scale_batch(self, batch, scale):
        reshaped_dims = tuple([int(scale*x) for x in list((batch.size(2), batch.size(3)))])
        scaled = torch.zeros((batch.size(0), batch.size(1), *reshaped_dims))
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i]), reshaped_dims)
            scaled[i] = transforms.ToTensor()(img)
        if reshaped_dims < self.img_dims[1:]:
            x, y = tuple([(x-x2) for (x, x2) in zip(self.img_dims[1:], reshaped_dims)])
            scaled = F.pad(scaled, (x, 0, y, 0))
        return scaled

    def _rotate_batch(self, batch, theta):
        rotated = torch.zeros_like(batch)
        for i in range(batch.size(0)):
            img = TF.rotate(transforms.ToPILImage()(batch[i]), theta)
            rotated[i] = transforms.ToTensor()(img)
        return rotated

    def _scale_rotate_batch(self, batch, scale, theta):
        scaled_batch = self._scale_batch(batch, scale)
        return self._rotate_batch(scaled_batch, theta)

    def preprocess_batch(self, x, batch_start_idx=None, batch_end_idx=None):
        if self.rotations and self.scaling:
            self._generate_angles()
            self._generate_scales()
            x_t = self._scale_rotate_batch(x, self.scale_1, self.theta_1)
            x_next = self._scale_rotate_batch(x, self.scale_2, self.theta_2)
        elif self.scaling:
            self._generate_scales()
            x_t = self._scale_batch(x, self.scale_1)
            x_next = self._scale_batch(x, self.scale_2)
        elif self.rotations:
            self._generate_angles()
            x_t = self._rotate_batch(x, self.theta_1)
            x_next = self._rotate_batch(x, self.theta_2)
        else:
            raise ValueError("Prepro of batch failed")
        if batch_start_idx is not None and batch_end_idx is not None:
            self._save_params(batch_start_idx, batch_end_idx)
        return x_t, x_next

class DeterministicPreprocessing():
    def __init__(self, num_test_samples, img_dims, num_rotations, num_scales, theta_range, scale_range):
        self.num_test_samples = num_test_samples
        self.img_dims = img_dims
        self.prepro_params = {}
        if num_scales <= 0 or num_rotations <= 0:
            raise ValueError("Det. prepro failed because rotations and/or scales should be > 0")
        if num_rotations > 0:
            self.thetas = np.linspace(0, 360, num_rotations)
            # check range is >= [0] and <= [1]
            if not self.thetas[0] >= theta_range[0] or not self.thetas[1] <= theta_range[1]:
                raise ValueError("Theta range does not match the trained range. Trained: {}, Prepro: {}".format(theta_range, self.thetas))
            self.theta_1, self.theta_2 = 0, 0
            self.prepro_params["theta_1"] = np.zeros((num_rotations, num_test_samples))
            self.prepro_params["theta_diff"] = np.zeros((num_rotations, num_test_samples))
        if num_scales > 0:
            self.scales = 0.7 + np.linspace(0, 1, num_scales) * 0.6
            if not self.scales[0] >= scale_range[0] or not self.scales[1] <= scale_range[1]:
                raise ValueError("Scale range does not match the trained range. Trained: {}, Prepro: {}".format(scale_range, self.scales))
            self.scale_1, self.scale_2 = 0.0, 0.0
            self.prepro_params["scale_1"] = np.zeros((num_scales, num_test_samples))
            self.prepro_params["scale_diff"] = np.zeros((num_scales, num_test_samples))

    # check if how much scaled shape will be <= self.data_loader.img_dims and then pad accordingliy
    def _scale_batch(self, batch, scale):
        reshaped_dims = tuple([int(scale*x) for x in list((batch.size(2), batch.size(3)))])
        scaled = torch.zeros((batch.size(0), batch.size(1), *reshaped_dims))
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i]), reshaped_dims)
            scaled[i] = transforms.ToTensor()(img)
        if reshaped_dims < self.img_dims[1:]:
            x, y = tuple([(x-x2) for (x, x2) in zip(self.img_dims[1:], reshaped_dims)])
            scaled = F.pad(scaled, (x, 0, y, 0))
        return scaled

    def _rotate_batch(self, batch, theta):
        rotated = torch.zeros_like(batch)
        for i in range(batch.size(0)):
            img = TF.rotate(transforms.ToPILImage()(batch[i]), theta)
            rotated[i] = transforms.ToTensor()(img)
        return rotated

    def _scale_rotate_batch(self, batch, scale, theta):
        scaled_batch = self._scale_batch(batch, scale)
        return self._rotate_batch(scaled_batch, theta)

    def preprocess_batch(self, x, scale, theta):
        x_t = self._scale_rotate_batch(x, scale, theta)
        x_next = self._scale_rotate_batch(x, scale, theta)
        return x_t, x_next