import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW
from samplers import ClassSampler, SingleDataPointSampler
from transforms import Rotate, Scale, CustomToPILImage

class DataLoader():
    def __init__(self, directories, batch_size, dataset, thetas=None, scales=None, single_x=False, specific_class=None):
        self.directories = directories
        self.n_classes = None
        self.c = None
        self.h = None
        self.w = None
        self.img_dims = None
        self.batch_size = batch_size
        self.dataset = dataset
        root = directories.data_dir_prefix+dataset
        self.thetas = thetas
        self.scales = scales
        self.scale_obj = None
        self.rotate_obj = None
        self.prepro_params = {}
        
        if dataset == "MNIST":
            self.n_classes = 10
            self.c = 1
            self.h = 28
            self.w = 28
            self.img_dims = (self.h, self.w)
            if self.thetas or self.scales:
                self._init_transforms()
            train_set = datasets.MNIST(root=root, train=True, transform=self._get_transforms(batch_size, train=True), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=self._get_transforms(batch_size, train=False), download=True)
        elif dataset == "LFW":
            self.c = 1
            data = DatasetLFW(root)
            self.h = data.h
            self.w = data.w
            self.img_dims = (self.h, self.w)
            self.n_classes = data.num_classes
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        elif dataset == "FF":
            self.c = 1
            self.h = 28
            self.w = 20
            self.img_dims = (self.h, self.w)
            data = DatasetFF(root)
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        else:
            raise ValueError("DATASET N/A!")
        # TODO: include the channel in dims and adjust other code accordingly
        self.input_dim = np.prod(self.img_dims)
        self.with_labels = dataset != "FF"
        self.single_x = single_x
        self.specific_class = specific_class
        self._set_data_loader(train_set, test_set, batch_size)
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)
        self.num_test_samples = 0 if self.single_x and not self.specific_class\
                                else len(self.test_loader.dataset)

    # TODO: Should for training/testing the ranges be different?
    def _init_transforms(self):
        if self.thetas:
            self.theta_range_1, self.theta_range_2 = [v for _, v in self.thetas.items()]
            self.theta_range_1[1] += 1
            self.theta_range_2[1] += 1
            self.prepro_params["theta_1"] = []
            self.prepro_params["theta_diff"] = []
        if self.scales:
            # find the max possible scale and set img dims to be that
            self.scale_range_1, self.scale_range_2 = self.scales["scale_1"], self.scales["scale_2"]
            max_scale = round(self.scale_range_1[1] + self.scale_range_2[1], 1)
            self.img_dims = tuple([int(max_scale*x) for x in list(self.img_dims)])
            self.input_dim = np.prod(self.img_dims)
            self.prepro_params["scale_1"] = []
            self.prepro_params["scale_diff"] = []

    def _get_transforms(self, batch_size, train=True):
        rotate = self.thetas is not None
        scale = self.scales is not None
        if rotate and not scale:
            if train:
                self.rotate_obj = Rotate(batch_size, self.theta_range_1, self.theta_range_2, self.prepro_params)
                return self.rotate_obj
            else:
                rotate_obj = Rotate(batch_size, self.theta_range_1, self.theta_range_2)
                return rotate_obj
        if scale and not rotate:
            if train:
                self.scale_obj = Scale(batch_size, self.img_dims, self.scale_range_1, self.scale_range_2, self.prepro_params)
                return self.scale_obj
            else:
                scale_obj = Scale(batch_size, self.img_dims, self.scale_range_1, self.scale_range_2)
                return scale_obj
        if scale and rotate:
            if train:
                self.scale_obj = Scale(batch_size, self.img_dims, self.scale_range_1, self.scale_range_2, self.prepro_params)
                self.rotate_obj = Rotate(batch_size, self.theta_range_1, self.theta_range_2, self.prepro_params)
                return transforms.Compose([self.scale_obj, CustomToPILImage(), self.rotate_obj])
            else:
                scale_obj = Scale(batch_size, self.img_dims, self.scale_range_1, self.scale_range_2)
                rotate_obj = Rotate(batch_size, self.theta_range_1, self.theta_range_2)
                return transforms.Compose([scale_obj, CustomToPILImage(), rotate_obj])
        else:
            return transforms.ToTensor()

    def _set_data_loader(self, train_set, test_set, batch_size):
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        if self.single_x and self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=batch_size, sampler=ClassSampler(train_set, self.specific_class, True),\
                shuffle=False, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=batch_size, sampler=ClassSampler(test_set, self.specific_class, True),\
                shuffle=False, **kwargs)
        elif self.single_x:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=batch_size, sampler=SingleDataPointSampler(train_set), drop_last=True,\
                shuffle=False, **kwargs)
        elif self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=batch_size, sampler=ClassSampler(train_set, self.specific_class),\
                drop_last=True, shuffle=False, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=batch_size, sampler=ClassSampler(test_set, self.specific_class),\
                drop_last=True, shuffle=False, **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)