import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import scipy.io

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

class DataLoader():
    
    def __init__(self, path, batch_size, dataset, z_dim):
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_set = None
        test_set = None
        data_folder_prefix = "../data/"

        self.data = None
        self.n_classes = None
        self.h = None
        self.w = None

        self.dataset = dataset
        self.folder_name = path + "/" + dataset + "_z=" + str(z_dim)
        
        if dataset == "MNIST":
            train_set = datasets.MNIST(root=data_folder_prefix, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=data_folder_prefix, train=False, transform=transforms.ToTensor(), download=False)
            self.n_classes = 10
            self.h = 28
            self.w = 28
        elif dataset == "EMNIST": # https://www.westernsydney.edu.au/__data/assets/text_file/0019/1204408/EMNIST_Readme.txt
            train_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=False, transform=transforms.ToTensor(), download=False)
        elif dataset == "LFW":
            lfw = fetch_lfw_people(data_home=data_folder_prefix+"/LFW", resize=0.4)
            _, h, w = lfw['images'].shape
            X = lfw['data']
            y = lfw['target']
            target_names = lfw['target_names']
            self.n_classes = target_names.shape[0]
            # split into a training and testing set
            train_set, test_set, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42) # TODO random_state=42
            self.data = train_set # train_set in numpy
            self.h = 50
            self.w = 37
            #print("Total dataset size:")
            #print("n_samples: %d" % n_samples)
            #print("n_features: %d" % X.shape[1])
            #print("n_classes: %d" % n_classes)
            #print("img dims x: {} y: {}".format(h, w)) # 50 x 37
            # transform data
            train_set = self.prepare_data_set(train_set, y_train, h, w)
            test_set = self.prepare_data_set(test_set, y_test, h, w)
        elif dataset == "FF":
            #print("HAHA")
            self.h = 28
            self.w = 20
            ff = scipy.io.loadmat(data_folder_prefix+"/FF/frey_rawface.mat")
            #print("test")
            ff = ff["ff"].T.reshape((-1, 1, self.h, self.w))
            #print(ff.shape)
            ff = ff.astype('float32')/255.
            self.data = ff
            #print(ff)
            #print(ff)
            train_set = torch.from_numpy(ff)
            test_set = torch.from_numpy(ff)
            #print(ff_torch.shape)
        else:
            print("DATASET N/A!")
            sys.exit()
        
        self.img_dims = (self.h, self.w)
        self.input_dim = np.prod(self.img_dims)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, **kwargs)

    def prepare_data_set(self, X, y, h, w):
        y_tensor = torch.FloatTensor(y)
        X = X/255.0
        X = torch.stack([torch.FloatTensor(i) for i in X])
        x_tensor = X.view(X.size(0), 1, h, w)
        data_set = [(x, y) for (x, y) in zip(x_tensor, y_tensor)]
        return data_set