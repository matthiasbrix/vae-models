import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from datasets import DatasetFF, DatasetLFW, DatasetLungScans
from samplers import ClassSampler, SingleDataPointSampler
from transforms import Rotate, Scale, CustomToPILImage

class DataLoader():
    def __init__(self, directories, batch_size, dataset, num_generations=1, thetas=None, scales=None, single_x=False,\
        specific_class=None, resize=None, fixed_thetas=None):
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
        self.num_generations = num_generations
        self.fixed_thetas = np.linspace(-180, 180, self.num_generations) if fixed_thetas else None
        self.prepro_params = {}
        root = directories.data_dir_prefix+dataset

        if dataset == "MNIST":
            self.n_classes = 10
            self.img_dims = (self.c, self.h, self.w) = (1, 28, 28)
            if self.thetas or self.scales:
                self._prepare_transforms()
            train_set = datasets.MNIST(root=root, train=True, transform=self._init_transform(), download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=self._init_transform(), download=True)            
        elif dataset == "LFW":
            data = DatasetLFW(root)
            self.c = 1
            self.h = data.h
            self.w = data.w
            self.img_dims = (self.c, self.h, self.w)
            self.n_classes = data.num_classes
            train_set, test_set = self._split_dataset(data)
        elif dataset == "FF":
            data = DatasetFF(root)
            self.c = 1
            self.h = 28
            self.w = 20
            self.img_dims = (self.c, self.h, self.w)
            train_set, test_set = self._split_dataset(data)
        elif dataset == "SVHN":
            self.n_classes = 10
            self.c = 3
            self.h = 32
            self.w = 32
            self.img_dims = (self.c, self.h, self.w)
            train_set = datasets.SVHN(root=root, split="train", transform=transforms.ToTensor(), download=True)
            test_set = datasets.SVHN(root=root, split="test", transform=transforms.ToTensor(), download=True)
        elif dataset == "LungScans":
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

        self.input_dim = np.prod(self.img_dims) # TODO: call num_pixels
        self.with_labels = dataset not in ["FF", "LungScans"]
        self.single_x = single_x
        self.specific_class = specific_class
        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = self.num_train_batches*self.batch_size
        self.num_test_samples = 0 if self.single_x and not self.specific_class\
                                else len(self.test_loader.dataset)

        if self.thetas:
            self.prepro_params["theta_1"] = np.zeros((self.num_generations, self.num_train_samples))
            self.prepro_params["theta_diff"] = np.zeros((self.num_generations, self.num_train_samples))
        if self.scales:
            self.prepro_params["scale_1"] = np.zeros((self.num_generations, self.num_train_samples))
            self.prepro_params["scale_diff"] = np.zeros((self.num_generations, self.num_train_samples))

    def _prepare_transforms(self):
        if self.thetas:
            self.theta_range_1, self.theta_range_2 = [v for _, v in self.thetas.items()]
            self.theta_range_1[1] += 1
            self.theta_range_2[1] += 1
        if self.scales:
            # find the max possible scale and set img dims to be that
            self.scale_range_1, self.scale_range_2 = self.scales["scale_1"], self.scales["scale_2"]
            max_scale = round(self.scale_range_1[1] + self.scale_range_2[1], 1)
            new_spatial_dims = tuple([int(max_scale*x) for x in list(self.img_dims[1:])])
            self.img_dims = (self.c, *new_spatial_dims)
            self.input_dim = np.prod(self.img_dims)

    def _init_transform(self):
        if self.thetas and not self.scales:
            return Rotate(self.batch_size, self.theta_range_1, self.theta_range_2, self.fixed_thetas, self.num_generations)
        if self.scales and not self.thetas:
            return Scale(self.batch_size, self.img_dims, self.scale_range_1, self.scale_range_2)
        if self.scales and self.thetas:
            scale_obj = Scale(self.batch_size, self.img_dims, self.scale_range_1, self.scale_range_2)
            rotate_obj = Rotate(self.batch_size, self.theta_range_1, self.theta_range_2, self.fixed_thetas, self.num_generations)
            return transforms.Compose([scale_obj, CustomToPILImage(), rotate_obj])
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

    def get_current_transform(self):
        # for datasets not called from torchvision.dataset and packed in a compose (LungScans)
        if self.data:
            return self.data.transform.transforms[-1]
        elif self.dataset == "MNIST":
            if self.thetas and self.scales:
                # save for thetas and scales that are in a Compose object
                return self.train_loader.dataset.transform.transforms
            else:
                # either scales or thetas
                return self.train_loader.dataset.transform
        else:
            raise ValueError("TRANSFORM OBJ N/A!")

    def save_prepro_params(self, batch_start_idx, batch_end_idx):
        if self.data:
            self.get_current_transform().save_params(self.prepro_params, batch_start_idx, batch_end_idx)
        elif self.dataset == "MNIST":
            if self.thetas and self.scales:
                t = self.get_current_transform()
                t[0].save_params(self.prepro_params, batch_start_idx, batch_end_idx)
                t[-1].save_params(self.prepro_params, batch_start_idx, batch_end_idx)
            else:
                 self.get_current_transform().save_params(self.prepro_params, batch_start_idx, batch_end_idx)
        else:
            raise ValueError("SAVE OF PARAMETERS N/A!")

    def signal_transform_last_epoch(self):
        if self.data:
            self.get_current_transform().set_fixed_theta()
        elif self.dataset == "MNIST":
            if self.thetas and self.scales:
                t = self.get_current_transform()
                t[0].set_fixed_theta()
                t[-1].set_fixed_theta()
            else:
                self.get_current_transform().set_fixed_theta()
        else:
            raise ValueError("SIGNAL N/A!")

