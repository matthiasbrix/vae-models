import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW, DatasetLungScans
from transforms import Rotate, Scale, ScaleRotate
from samplers import SingleDataPointSampler, ClassSampler

# Set in model params thetas and scales to None and trigger default params for transofmraitons
# To have only one of the modes do
class DataLoader():
    def __init__(self, directories, batch_size, dataset, thetas=None, scales=None, resize=None):
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
            train_set = datasets.MNIST(root=root, train=True, transform=self._init_transform(), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=self._init_transform(), download=True)
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
        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_test_batches = len(self.test_loader)
        # could also call len(self.train_loader.dataset) but is more flexible this way
        self.num_train_samples = self.num_train_batches*self.batch_size
        self.num_test_samples = self.num_test_batches*self.batch_size
        
        self.dataset = dataset
        self.root = root

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

    # Even though we use tdcvae as model and pytorch transforms,
    # we don't apply the transform like in training as we do this
    # explicitly in preprocessing to save the parameters conveniently.
    def get_new_test_data_loader(self, sampler=None):
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        else:
            _, test_set = self._split_dataset(self.data)
        if sampler is not None and sampler[0] == "single_point":
            sampler = SingleDataPointSampler(test_set, sampler[1])
        if sampler is not None and sampler[0] == "specific_class":
            sampler = ClassSampler(test_set, sampler[2], sampler[1])
        batch_size = 1 if sampler is not None else self.batch_size
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)