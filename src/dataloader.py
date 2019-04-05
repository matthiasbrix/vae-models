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

        if dataset == "MNIST":
            train_set = datasets.MNIST(root=data_folder_prefix, train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(root=data_folder_prefix, train=False, transform=transforms.ToTensor(), download=False)
        elif dataset == "EMNIST": # https://www.westernsydney.edu.au/__data/assets/text_file/0019/1204408/EMNIST_Readme.txt
            train_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.EMNIST(root=data_folder_prefix, split="balanced", train=False, transform=transforms.ToTensor(), download=False)
        elif dataset == "LFW":
            lfw = fetch_lfw_people(data_home=data_folder_prefix+"/LFW", resize=0.4) # min_faces_per_person=70, 
            n_samples, h, w = lfw['images'].shape # c 
            X = lfw['data']
            n_features = X.shape[1]
            y = lfw['target']
            target_names = lfw['target_names']
            n_classes = target_names.shape[0]     
            # split into a training and testing set, labels are ignored
            train_set, test_set, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42) # TODO remove later random state
            
            print("Total dataset size:")
            print("n_samples: %d" % n_samples)
            print("n_features: %d" % n_features)
            print("n_classes: %d" % n_classes)
            print("img dims x: {} y: {}".format(h, w)) # 50 x 37 x 3
            
            train_set = train_set/255.0
            data_transform = transforms.Compose([transforms.ToTensor()])
            train_set = data_transform(train_set) # tensor 1, 1030, 5550
            y_tensor = torch.tensor(y_train, dtype=torch.long)
            #y_tensor = y_tensor.view(y_tensor.size(0), 1)
            train_set = train_set.view(train_set.size(1), 1, h, w) # 1030 x 1 x 50 x 37 x 3
            # TODO: need list of [(data, target)] - try with tensor cat([])
            print(train_set.shape, y_tensor.shape)
            ls = []
            for i in range(train_set.shape[0]):
                c = (train_set[i], y_tensor[i])
                ls.append(c)
            #print(ls)
            train_set = ls

            ls = []
            test_set = test_set/255.0
            test_set = data_transform(test_set) # tensor 1, 1030, 5550
            y_tensor = torch.tensor(y_test, dtype=torch.long)
            test_set = test_set.view(test_set.size(1), 1, h, w) # 1030 x 1 x 50 x 37 x 3
            for i in range(test_set.shape[0]):
                c = (test_set[i], y_tensor[i])
                ls.append(c)
            test_set = ls
            #print("kaeft", torch.tensor(y_train).shape)
            #train_set = (train_set.type(torch.LongTensor), y_tensor) # TODO: want 1030 x (1 x 50 x 37 x 3, 1), have 1030 x 1 x 50 x 37 x 3, 1030 x 1
            # TODO test set to tensor and targets
        else:
            print("DATASET N/A!")
            sys.exit()
        
        #print(train_set)
        # TODO
        # MNIST has 60k x 1 x 28 x 28 in data
        # LFW has 1 x 1030 x 5550 but want 1030 x 1 x 50 x 37 x 3
        #print(len(train_set), train_set.shape) # 
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, **kwargs)
        self.folder_name = path + "/" + dataset + "_z=" + str(z_dim)

        #print(train_set)
        #print(self.train_loader.dataset)
        #for i, a in enumerate(train_set):
        #    print(i, a)
