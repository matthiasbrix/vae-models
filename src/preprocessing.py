from random import uniform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import numpy as np

class Preprocessing():
    def __init__(self, data_loader, thetas=None, scales=None):
        self.rotate = thetas is not None
        self.scale = scales is not None
        self.data_loader = data_loader
        self.prepro_params = {}
        if self.rotate:
            self.theta_range_1, self.theta_range_2 = [v for _, v in thetas.items()]
            # add 1's because it's exclusive when gen. random
            self.theta_range_1[1] += 1
            self.theta_range_2[1] += 1
            self.prepro_params["theta_1"] = []
            self.prepro_params["theta_diff"] = []
        if self.scale:
            # find the max possible scale and set img dims to be that
            self.scale_range_1, self.scale_range_2 = scales["scale_1"], scales["scale_2"]
            max_scale = round(self.scale_range_1[1] + self.scale_range_2[1], 1)
            self.data_loader.img_dims = tuple([int(max_scale*x) for x in list(self.data_loader.img_dims)])
            self.data_loader.input_dim = np.prod(self.data_loader.img_dims)
            self.prepro_params["scale_1"] = []
            self.prepro_params["scale_diff"] = []
        self.theta_1, self.theta_2, self.scale_1, self.scale_2 = None, None, None, None

    def _generate_angles(self):
        theta_1 = np.random.randint(*self.theta_range_1)
        theta_2 = theta_1 + np.random.randint(*self.theta_range_2)
        return theta_1, theta_2

    def _generate_scales(self):
        scale_1 = uniform(*self.scale_range_1)
        scale_2 = scale_1 + uniform(*self.scale_range_2)
        if scale_1 <= 0 or scale_2 <= 0:
            raise ValueError("One of the scales is <= 0!")
        return scale_1, scale_2

    def _rotate_batch(self, batch, angle):
        rotated = torch.zeros_like(batch)
        for i in range(batch.size(0)):
            img = TF.rotate(transforms.ToPILImage()(batch[i]), angle)
            rotated[i] = transforms.ToTensor()(img)
        return rotated

    # check if how much scaled shape will be <= self.data_loader.img_dims and then pad accordingliy
    def _scale_batch(self, batch, scale):
        reshaped_dims = tuple([int(scale*x) for x in list((batch.size(2), batch.size(3)))])
        scaled = torch.zeros((batch.size(0), batch.size(1), *reshaped_dims))
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i]), reshaped_dims)
            scaled[i] = transforms.ToTensor()(img)
        if reshaped_dims < self.data_loader.img_dims:
            x, y = tuple([(x-x2) for (x, x2) in zip(self.data_loader.img_dims, reshaped_dims)])
            scaled = F.pad(scaled, (x, 0, y, 0))
        return scaled

    def _scale_rotate_batch(self, batch, scale, angle):
        scaled_batch = self._scale_batch(batch, scale)
        return self._rotate_batch(scaled_batch, angle)

    def save_params(self):
        if self.rotate:
            theta_diff = self.theta_2 - self.theta_1
            self.prepro_params["theta_diff"].append(theta_diff)
            self.prepro_params["theta_1"].append(self.theta_1)
        if self.scale:
            scale_diff = self.scale_2 - self.scale_1
            self.prepro_params["scale_diff"].append(scale_diff)
            self.prepro_params["scale_1"].append(self.scale_1)

    def preprocess_batch(self, x):
        if self.rotate and self.scale:
            self.theta_1, self.theta_2 = self._generate_angles()
            self.scale_1, self.scale_2 = self._generate_scales()
            x1 = self._scale_rotate_batch(x, self.scale_1, self.theta_1)
            x2 = self._scale_rotate_batch(x, self.scale_2, self.theta_2)
        elif self.scale:
            self.scale_1, self.scale_2 = self._generate_scales()
            x1 = self._scale_batch(x, self.scale_1)
            x2 = self._scale_batch(x, self.scale_2)
        elif self.rotate:
            self.theta_1, self.theta_2 = self._generate_angles()
            x1 = self._rotate_batch(x, self.theta_1)
            x2 = self._rotate_batch(x, self.theta_2)
        else:
            raise ValueError("Prepro of batch failed")
        return x1.view(-1, self.data_loader.input_dim), x2.view(-1, self.data_loader.input_dim)