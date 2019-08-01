import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW, DatasetLungScans
from samplers import ClassSampler, SingleDataPointSampler
from transforms import Rotate, Scale, ScaleRotate

class DataLoader():
    def __init__(self, directories, batch_size, dataset, thetas=None, scales=None, single_x=False,\
        specific_class=None, resize=None):
        self.directories = directories
        self.data = None
        self.n_classes = None
        self.c = None
        self.h = None
        self.w = None
        self.img_dims = None
        self.batch_size = batch_size
        self.dataset = dataset
        self.thetas = thetas
        self.scales = scales

        root = directories.data_dir_prefix+dataset

        if dataset.lower() == "mnist":
            self.n_classes = 10
            self.img_dims = (self.c, self.h, self.w) = (1, 28, 28)
            if self.thetas or self.scales:
                self._prepare_transforms()
            train_set = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
            #train_set = datasets.MNIST(root=root, train=True, transform=self._init_transform(), download=True)
            #test_set = datasets.MNIST(root=root, train=False, transform=self._init_transform(), download=True)
        elif dataset == "lfw":
            self.data = DatasetLFW(root)
            self.c = 1
            self.h = self.data.h
            self.w = self.data.w
            self.img_dims = (self.c, self.h, self.w)
            self.n_classes = self.data.num_classes
            train_set, test_set = self._split_dataset(self.data)
        elif dataset == "ff":
            self.data = DatasetFF(root)
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
            if self.thetas or self.scales:
                self._prepare_transforms()
            # reading all the sets of images that are specified in the list
            folders = ["/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.52915333682423613339719948113721836450_OBICone-beamCT/",
                       "/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.55302824863178429077114755927787508155_OBICone-beamCT/",
                       "/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.542181959870340811013566519894670057885_OBICone-beamCT/"]
            if resize:
                transform = transforms.Compose([
                    transforms.Resize(resize),
                    self._init_transform()
                ])
            else:
                transform = None
            self.data = DatasetLungScans(root, folders, transform)
            self.img_dims = (self.c, *resize)
            train_set, test_set = self._split_dataset(self.data)
        else:
            raise ValueError("DATASET N/A!")

        self.input_dim = np.prod(self.img_dims)
        self.with_labels = dataset not in ["ff", "lungscans"]
        self.single_x = single_x
        self.specific_class = specific_class
        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_test_batches = len(self.test_loader)
        self.num_train_samples = self.num_train_batches*self.batch_size
        self.num_test_samples = 0 if self.single_x and not self.specific_class\
                                else len(self.test_loader.dataset)
        
        self.dataset = dataset
        self.root = root

        print(self.num_test_batches, self.num_train_batches)

    def _prepare_transforms(self):
        # adjust for that the uniform ranges are excludsive
        if self.thetas:
            self.theta_range_1, self.theta_range_2 = self.thetas["theta_1"], self.thetas["theta_2"]
        if self.scales:
            self.scale_range_1, self.scale_range_2 = self.scales["scale_1"], self.scales["scale_2"]

    def _init_transform(self):
        if self.thetas and not self.scales:
            print("rotate")
            return Rotate(self.batch_size, self.theta_range_1, self.theta_range_2)
        if self.scales and not self.thetas:
            print("scale")
            return Scale(self.batch_size, self.scale_range_1, self.scale_range_2)
        if self.scales and self.thetas:
            print("scalerotate")
            return ScaleRotate(self.batch_size, self.scale_range_1, self.scale_range_2, self.theta_range_1, self.theta_range_2)
        else:
            return transforms.ToTensor()

    def _set_data_loader(self, train_set, test_set):
        if self.single_x and self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=ClassSampler(train_set, self.specific_class, True),\
                shuffle=False)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, sampler=ClassSampler(test_set, self.specific_class, True),\
                shuffle=False)
        elif self.single_x:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=SingleDataPointSampler(train_set), drop_last=True,\
                shuffle=False)
        elif self.specific_class:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, sampler=ClassSampler(train_set, self.specific_class),\
                drop_last=True, shuffle=False)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, sampler=ClassSampler(test_set, self.specific_class),\
                drop_last=True, shuffle=False)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True)

    def _split_dataset(self, data):
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        return torch.utils.data.random_split(data, [train_size, test_size])

    # Even though we use tdcvae as model, we don't apply the transform as we do this explicitly in preprocessing
    # to save the parameters.
    def get_new_test_data_loader(self):
        if self.dataset == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        else:
            _, test_set = self._split_dataset(self.data)
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, drop_last=True, shuffle=True)

