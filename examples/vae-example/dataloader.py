import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

class DataLoader():
    
    def __init__(self, batch_size, dataset, z_dim):
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_set = None
        test_set = None
	data_folder_prefix = "../../data"

        if dataset == "MNIST":
            train_set = datasets.MNIST(root=data_folder_prefix, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=data_folder_prefix, train=False, transform=transforms.ToTensor(), download=False)
        elif dataset == "EMNIST": # https://www.westernsydney.edu.au/__data/assets/text_file/0019/1204408/EMNIST_Readme.txt
            train_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=False, transform=transforms.ToTensor(), download=False)
        elif dataset == "LFW":
            lfw = fetch_lfw_people(data_home=data_folder_prefix+"/LFW", min_faces_per_person=70, color=True, resize=0.4)
            n_samples, h, w, c = lfw['images'].shape
            X = lfw['data']
            n_features = X.shape[1]
            y = lfw['target']
            target_names = lfw['target_names']
            n_classes = target_names.shape[0]     
            # split into a training and testing set, labels are ignored
            train_set, test_set, _, _ = train_test_split(
                X, y, test_size=0.20, random_state=42)
            
            print("Total dataset size:")
            print("n_samples: %d" % n_samples)
            print("n_features: %d" % n_features)
            print("n_classes: %d" % n_classes)
            print("img dims x: {} y: {} c: {}".format(h, w, c)) # 50 x 37 x 3
            print(train_set.shape)

            data_transform = transforms.Compose([transforms.ToTensor()])
            train_set = data_transform(train_set)
        else:
            print("DATASET N/A!")
            sys.exit()
        
        # MNIST has tensor 1, 28, 28, LFW 1, 1030, 5550
        print((train_set.size()))
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.folder_name = dataset + "_z=" + str(z_dim)
