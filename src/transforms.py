from random import uniform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

class Rotate(object):
    def __init__(self, batch_size, theta_range_1, theta_range_2, time_agnostic):
        self.count = 0
        self.batch_size = batch_size
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0
        self.theta_2 = 0
        self.time_agnostic = time_agnostic

    # is called data point wise, hence need to count for batch to apply one
    # set of angles to a batch
    def __call__(self, sample):
        # for scale + rotate mode (because we used CustomToPILImage to pack x_t, x_next into a tuple)
        if isinstance(sample, tuple):
            x_t, x_next = sample
        else:
            x_t = sample
            x_next = sample
        if self.count % self.batch_size == 0:
            self._generate_angles()
        x_t = TF.rotate(x_t, self.theta_1)
        x_next = TF.rotate(x_next, self.theta_2)
        self.count += 1
        return transforms.ToTensor()(x_t), transforms.ToTensor()(x_next)

    def _generate_angles(self):
        self.theta_1 = np.random.randint(*self.theta_range_1)
        self.theta_2 = self.theta_1 + np.random.randint(*self.theta_range_2)

class Scale(object):
    def __init__(self, batch_size, img_dims, scale_range_1, scale_range_2):
        self.count = 0
        self.batch_size = batch_size
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.img_dims = img_dims

    def __call__(self, sample):
        if self.count % self.batch_size == 0:
            self._generate_scales()
        x_t = self._scale_sample(self.scale_1, sample)
        x_next = self._scale_sample(self.scale_2, sample)
        self.count += 1
        return x_t, x_next

    def _generate_scales(self):
        self.scale_1 = round(uniform(*self.scale_range_1), 2)
        self.scale_2 = round(self.scale_1 + uniform(*self.scale_range_2), 2)
        if self.scale_1 <= 0 or self.scale_2 <= 0:
            raise ValueError("One of the scales is <= 0!")
    
    # scaling of a single smaple (C, H, W), but excluding the scaling of the channel
    def _scale_sample(self, scale, sample):
        sample_tensor = transforms.ToTensor()(sample)
        reshaped_spatial_dims = tuple([int(scale*x) for x in list((sample_tensor.size(1), sample_tensor.size(2)))])
        scaled_sample = transforms.ToTensor()(TF.resize(sample, reshaped_spatial_dims))
        if reshaped_spatial_dims < self.img_dims[1:]:
            x, y = tuple([(x-x2) for (x, x2) in zip(self.img_dims[1:], reshaped_spatial_dims)])
            scaled_sample = F.pad(scaled_sample, (x, 0, y, 0))
        return scaled_sample

class CustomToPILImage(object):
    def __call__(self, sample_pair):
        x_t, x_next = sample_pair
        return transforms.ToPILImage()(x_t), transforms.ToPILImage()(x_next)