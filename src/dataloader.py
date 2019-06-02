import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW
from samplers import ClassSampler, SingleDataPointSampler

class DataLoader():
    def __init__(self, directories, batch_size, dataset, single_x=False, specific_class=None):
        self.directories = directories
        self.data = None
        self.n_classes = None
        self.c = None
        self.h = None
        self.w = None
        self.batch_size = batch_size
        self.dataset = dataset
        root = directories.data_dir_prefix+dataset
        if dataset == "MNIST":
            self.n_classes = 10
            self.c = 1
            self.h = 28
            self.w = 28
            train_set = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
        elif dataset == "LFW":
            self.c = 1
            data = DatasetLFW(root)
            self.h = data.h
            self.w = data.w
            self.n_classes = data.num_classes
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        elif dataset == "FF":
            self.c = 1
            self.h = 28
            self.w = 20
            data = DatasetFF(root)
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
        else:
            raise ValueError("DATASET N/A!")
        # TODO: include the channel and adjust other code accordingly
        self.img_dims = (self.h, self.w)
        self.input_dim = np.prod(self.img_dims)
        self.with_labels = dataset != "FF"
        self.single_x = single_x
        self.specific_class = specific_class
        self._set_data_loader(train_set, test_set, batch_size)
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = len(self.train_loader.dataset)
        self.num_test_samples = 0 if self.single_x and not self.specific_class\
                                else len(self.test_loader.dataset)

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