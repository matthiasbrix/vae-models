"""This module is for processing transformations on x_t, x_{t+1} during training of a TDCVAE model.
The transformations available are 1) rotation, 2) scaling and 3) rotation+scaling. The
module implements abstract classes by PyTorch, used for Dataset objects

"""
import torchvision.transforms as transforms
from preprocessing import preprocess_sample, generate_angles, generate_scales

class Rotate(object):
    """This class is for applying rotations on given ranges of angles

        Args:
            theta_range_1: A range of angles for x_t (if None, generate from default range)
            theta_range_2: A range of angles for x_{t+1} (if None, generate from default range)
    """
    def __init__(self, theta_range_1=None, theta_range_2=None):
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
        x_t = preprocess_sample(x_t, theta=self.theta_1)
        x_next = preprocess_sample(x_next, theta=self.theta_2)
        return x_t, x_next

class Scale(object):
    """This class is for applying scaling on given ranges of scales

        Args:
            scale_range_1: A range of scales for x_t (if None, generate from default range)
            scale_range_2: A range of scales for x_{t+1} (if None, generate from default range)
    """
    def __init__(self, scale_range_1=None, scale_range_2=None):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
        x_t = preprocess_sample(x_t, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, scale=(self.scale_2, self.scale_2))
        return x_t, x_next

class ScaleRotate(object):
    """This class is for applying rotations + scale on given ranges of scales and angles

        Args:
            scale_range_1: A range of scales for x_t (if None, generate from default range)
            scale_range_2: A range of scales for x_{t+1} (if None, generate from default range)
            theta_range_1: A range of angles for x_t (if None, generate from default range)
            theta_range_2: A range of angles for x_{t+1} (if None, generate from default range)
    """
    def __init__(self, scale_range_1=None, scale_range_2=None, theta_range_1=None, theta_range_2=None):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
        x_t = preprocess_sample(x_t, theta=self.theta_1, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, theta=self.theta_2, scale=(self.scale_2, self.scale_2))
        return x_t, x_next
