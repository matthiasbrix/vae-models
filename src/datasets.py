from torch.utils.data import Dataset
from sklearn.datasets import fetch_lfw_people
import scipy.io
import torch
import numpy as np
from load import load_all_volumes
import torchvision.transforms as transforms

class DatasetFF(Dataset):
    def __init__(self, file_path):
        c = 1
        h = 28
        w = 20
        ff = scipy.io.loadmat(file_path+"/frey_rawface.mat")
        ff = ff["ff"].T.reshape((-1, c, h, w))
        self.data = ff.astype("float32")/255.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DatasetLFW(Dataset):
    def __init__(self, file_path, resize=0.4):
        lfw = fetch_lfw_people(data_home=file_path, resize=resize)
        _, height, width = lfw['images'].shape
        self.h = height
        self.w = width
        self.num_classes = lfw['target_names'].shape[0]
        self.data = self._init_data_set(lfw['data'], lfw['target'], height, width)

    def _init_data_set(self, X, y, h, w):
        y_tensor = torch.FloatTensor(y)
        X = X/255.0
        X = torch.stack([torch.FloatTensor(i) for i in X])
        x_tensor = X.view(X.size(0), 1, h, w)
        data_set = [(x, y) for (x, y) in zip(x_tensor, y_tensor)]
        return data_set

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DatasetLungScans(Dataset):
    def __init__(self, file_path, volumes, transform=None):
        # prepend root to path
        volumes = [file_path+volume for volume in volumes]
        self.data = np.concatenate(tuple(load_all_volumes(volumes)), axis=0)
        # normalize to (0,1)
        self.data = (self.data - np.min(self.data))/np.ptp(self.data)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            image = torch.FloatTensor(np.expand_dims(self.data[idx], axis=0))
            return self.transform(transforms.ToPILImage()(image))
        return self.data[idx]