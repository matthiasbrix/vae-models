"""This module is for loading datasets, their processing and the PyTorch Dataloader.
It is also used to obtain test data set during test inference.

"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW, DatasetLungScans
from transforms import Rotate, Scale, ScaleRotate
from samplers import SingleDataPointSampler, ClassSampler

class DataLoader():
    """This class is a wrapper for everything related to loading and preprocessing the data.
    
        Args:
            directories: Directories class object.
            batch_size: The number of elements in a batch
            dataset: The dataset (as a string) to use corresponding to the list of the model function in model_params.py
            thetas: For TDCVAE, the thetas range in use for affine transformations.
            scales: For TDCVAE, the scales range in use for affine transformations.
            resize: For resizing images of lung scans for TDCVAE2 model
            crop: For cropping images of lung scans for TDCVAE2 model
            folders: For lung scans loading of dataset
        Note:
            Set in model params thetas and scales to None and trigger default params for transformations.
            To have only one of the modes, you have to insert the paremeters in the corresponding function
            in model_params.py.
    """
    def __init__(self, directories, batch_size, dataset, thetas=None, scales=None, resize=None, crop=None, folders=None):
        self.directories = directories
        self.data = None
        self.n_classes = None
        self.c = None
        self.h = None
        self.w = None
        self.img_dims = None
        self.batch_size = batch_size
        self.dataset = dataset
        # for temporal model (TDCVAE)
        self.thetas = thetas
        self.scales = scales
        # for lungscans
        self.folders = folders
        self.resize = resize
        self.crop = crop

        self.root = directories.data_dir_prefix+dataset

        if dataset.lower() == "mnist":
            self.n_classes = 10
            self.img_dims = (self.c, self.h, self.w) = (1, 28, 28)
            if self.thetas or self.scales:
                self._prepare_transforms()
            train_set = datasets.MNIST(root=self.root, train=True, transform=self._init_transform(), download=True)
            test_set = datasets.MNIST(root=self.root, train=False, transform=self._init_transform(), download=True)
        elif dataset == "lfw":
            self.data = DatasetLFW(self.root)
            self.c = 1
            self.h = self.data.h
            self.w = self.data.w
            self.img_dims = (self.c, self.h, self.w)
            self.n_classes = self.data.num_classes
            train_set, test_set = self._split_dataset(self.data)
        elif dataset == "ff":
            self.data = DatasetFF(self.root)
            self.c = 1
            self.h = 28
            self.w = 20
            self.img_dims = (self.c, self.h, self.w)
            train_set, test_set = self._split_dataset(self.data)
        elif dataset == "lungscans":
            self.c = 1
            self.h = 384
            self.w = 384
            self.img_dims = (self.c, self.h, self.w)
            self.data = DatasetLungScans(folders, resize, crop)
            if resize is not None:
                self.h = resize[0]
                self.w = resize[1]
                self.img_dims = (self.c, self.h, self.w)
            if crop is not None:
                h = crop[0][1] - crop[0][0] if crop[0] else self.h
                w = crop[1][1] - crop[1][0] if crop[1] else self.w
                self.img_dims = (self.c, h, w)
            train_set, test_set = self._split_dataset(self.data)
        else:
            raise ValueError("DATASET N/A!")

        self.input_dim = np.prod(self.img_dims)
        self.with_labels = dataset not in ["ff", "lungscans"]
        self.dataset = dataset

        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_test_batches = len(self.test_loader)
        # could also call len(self.train_loader.dataset) but is more flexible this way
        self.num_train_samples = self.num_train_batches*self.batch_size
        self.num_test_samples = self.num_test_batches*self.batch_size

    def _prepare_transforms(self):
        if self.thetas:
            self.theta_range_1, self.theta_range_2 = self.thetas["theta_1"], self.thetas["theta_2"]
        if self.scales:
            self.scale_range_1, self.scale_range_2 = self.scales["scale_1"], self.scales["scale_2"]

    def _init_transform(self):
        if self.thetas and not self.scales:
            return Rotate(self.theta_range_1, self.theta_range_2)
        if self.scales and not self.thetas:
            return Scale(self.scale_range_1, self.scale_range_2)
        if self.scales and self.thetas:
            return ScaleRotate(self.scale_range_1, self.scale_range_2, self.theta_range_1, self.theta_range_2)
        else:
            return transforms.ToTensor()

    def _set_data_loader(self, train_set, test_set):
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)

    def _split_dataset(self, data):
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        return torch.utils.data.random_split(data, [train_size, test_size])

    def get_new_test_data_loader(self, sampler=None):
        """Returns a dataloader with data from the test data set. If given a sampler
        we use that for the dataloader.

            Args:
                sampler: A tuple that in [0] element has a string denoting which sampler to use.

            Note:
                When we use tdcvae as model and pytorch transforms,
                we don't apply the transform like in training as we do this
                explicitly in preprocessing to save the parameters conveniently.
        """
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        elif self.dataset == "lungscans":
            data = DatasetLungScans(self.folders, self.resize, self.crop, sampling=True)
            self.data = data
            _, test_set = self._split_dataset(data)
        else:
            _, test_set = self._split_dataset(self.data)
        if sampler is not None and sampler[0] == "single_point":
            sampler = SingleDataPointSampler(test_set, sampler[1])
        elif sampler is not None and sampler[0] == "specific_class":
            sampler = ClassSampler(test_set, sampler[1], sampler[2])
        batch_size = 1 if sampler is not None else self.batch_size
        shuffle = sampler is None
        drop_last = sampler is None
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=drop_last)
