from random import uniform
from preprocessing import preprocess_sample
import torchvision.transforms as transforms
import numpy as np

class Rotate(object):
    def __init__(self, batch_size, theta_range_1, theta_range_2):
        self.count = 0
        self.batch_size = batch_size
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0
        self.theta_2 = 0

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
        x_t = preprocess_sample(x_t, theta=self.theta_1)
        x_next = preprocess_sample(x_next, theta=self.theta_2)
        self.count += 1
        return x_t, x_next

    def _generate_angles(self):
        self.theta_1 = np.random.randint(*self.theta_range_1)
        self.theta_2 = self.theta_1 + np.random.randint(*self.theta_range_2)

class Scale(object):
    def __init__(self, batch_size, scale_range_1, scale_range_2):
        self.count = 0
        self.batch_size = batch_size
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0

    def __call__(self, sample):
        x_t = sample
        x_next = sample
        if self.count % self.batch_size == 0:
            self._generate_scales()
        x_t = preprocess_sample(x_t, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, scale=(self.scale_2, self.scale_2))
        self.count += 1
        return x_t, x_next

    def _generate_scales(self):
        self.scale_1 = np.around(uniform(*self.scale_range_1), decimals=2)
        self.scale_2 = np.around(self.scale_1 + uniform(*self.scale_range_2), decimals=2)

class CustomToPILImage(object):
    def __call__(self, sample_pair):
        x_t, x_next = sample_pair
        return transforms.ToPILImage()(x_t), transforms.ToPILImage()(x_next)