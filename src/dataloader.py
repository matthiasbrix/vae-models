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
        self.thetas = thetas
        self.scales = scales
        self.prepro_params = {}
        root = directories.data_dir_prefix+dataset
        
        if dataset == "MNIST":
            self.n_classes = 10
            self.c = 1
            self.h = 28
            self.w = 28
            self.img_dims = (self.c, self.h, self.w)
            if self.thetas or self.scales:
                self._init_transforms()
            train_set = datasets.MNIST(root=root, train=True, transform=self._get_transform(), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=self._get_transform(), download=True)
        elif dataset == "LFW":
            self.c = 1
            data = DatasetLFW(root)
            self.h = data.h
            self.w = data.w
            self.img_dims = (self.c, self.h, self.w)
            self.n_classes = data.num_classes
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        elif dataset == "FF":
            self.c = 1
            self.h = 28
            self.w = 20
            self.img_dims = (self.c, self.h, self.w)
            data = DatasetFF(root)
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        elif dataset == "SVHN":
            self.n_classes = 10
            self.c = 3
            self.h = 32
            self.w = 32
            self.img_dims = (self.c, self.h, self.w)
            train_set = datasets.SVHN(root=root, split="train", transform=transforms.ToTensor(), download=True)
            test_set = datasets.SVHN(root=root, split="test", transform=transforms.ToTensor(), download=True)
        else:
            raise ValueError("DATASET N/A!")
        self.input_dim = np.prod(self.img_dims)
        self.with_labels = dataset != "FF"
        self.single_x = single_x
        self.specific_class = specific_class
        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)
        self.num_test_samples = 0 if self.single_x and not self.specific_class\
                                else len(self.test_loader.dataset)

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
            new_spatial_dims = tuple([int(max_scale*x) for x in list(self.img_dims[1:])])
            self.img_dims = (self.c, *new_spatial_dims)
            self.input_dim = np.prod(self.img_dims)
            self.prepro_params["scale_1"] = []
            self.prepro_params["scale_diff"] = []

    def _get_transform(self):
        if self.thetas and not self.scales:
            return Rotate(self.batch_size, self.theta_range_1, self.theta_range_2, self.prepro_params)
        if self.scales and not self.thetas:
            return Scale(self.batch_size, self.img_dims, self.scale_range_1, self.scale_range_2, self.prepro_params)
        if self.scales and self.thetas:
            scale_obj = Scale(self.batch_size, self.img_dims, self.scale_range_1, self.scale_range_2, self.prepro_params)
            rotate_obj = Rotate(self.batch_size, self.theta_range_1, self.theta_range_2, self.prepro_params)
            return transforms.Compose([scale_obj, CustomToPILImage(), rotate_obj])
        else:
            return transforms.ToTensor()

    def _set_data_loader(self, train_set, test_set):
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        if self.single_x and self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=ClassSampler(train_set, self.specific_class, True),\
                shuffle=False, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, sampler=ClassSampler(test_set, self.specific_class, True),\
                shuffle=False, **kwargs)
        elif self.single_x:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=SingleDataPointSampler(train_set), drop_last=True,\
                shuffle=False, **kwargs)
        elif self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=ClassSampler(train_set, self.specific_class),\
                drop_last=True, shuffle=False, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, sampler=ClassSampler(test_set, self.specific_class),\
                drop_last=True, shuffle=False, **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True, **kwargs)