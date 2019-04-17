import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
        self.dataset = dataset
        self.folder_name = path + "/" + dataset + "_z=" + str(z_dim)

        if dataset == "MNIST":
            train_set = datasets.MNIST(root=data_folder_prefix, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=data_folder_prefix, train=False, transform=transforms.ToTensor(), download=False)
            self.n_classes = 10
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
                X, y, test_size=0.20, random_state=42) # TODO remove later random state
            self.data = train_set # train_set in numpy
            #print("Total dataset size:")
            #print("n_samples: %d" % n_samples)
            #print("n_features: %d" % X.shape[1])
            #print("n_classes: %d" % n_classes)
            #print("img dims x: {} y: {}".format(h, w)) # 50 x 37
            # transform data
            train_set = self.prepare_data_set(train_set, y_train, h, w)
            test_set = self.prepare_data_set(test_set, y_test, h, w)
        else:
            print("DATASET N/A!")
            sys.exit()
        
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, **kwargs)

    def prepare_data_set(self, X, y, h, w):
        X = X/255.0
        data_transform = transforms.Compose([transforms.ToTensor()])
        X = data_transform(X) # tensor 1, 1030, 50*37
        y_tensor = torch.LongTensor(y)
        X_tensor = X.view(X.size(1), 1, h, w) # 1030 x 1 x 50 x 37
        data_set = [(x, y) for (x, y) in zip(X_tensor, y_tensor)]
        return data_set