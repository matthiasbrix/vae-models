from torch.utils.data import Dataset
from sklearn.datasets import fetch_lfw_people
import scipy.io
import torch
import numpy as np
from load import load_all_volumes

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
    def __init__(self, volumes, transform=None, resized_dims=None):
        self.data = np.concatenate(tuple(load_all_volumes(volumes)), axis=0) # shape is 192 x 384 x 384, as 192=64*3
        # normalize to (0,1)
        self.data = (self.data - np.min(self.data))/np.ptp(self.data)
        self.transform = transform
        self.resized_dims = resized_dims
        self.t0 = 32

    def __len__(self):
        return len(self.data)

    # TODO: maybe use t0 = 32, t1=33 because first and last are not so interesting...
    def __getitem__(self, t0):
        t1 = (t0+1) % self.data.shape[0] # x % 192 for lungscans
        print("timestamps", t0, t1)
        if self.transform is not None and self.resized_dims is not None:
            image1 = self.transform(self.data[t0], output_shape=self.resized_dims, anti_aliasing=True)
            image1 = image1[14:54, :]
            image1 = np.expand_dims(image1, axis=0) # inserts a channel dim at 0th index
            image2 = self.transform(self.data[t1+1], output_shape=self.resized_dims, anti_aliasing=True)
            image2 = np.expand_dims(image2, axis=0) # inserts a channel dim at 0th index
        else:
            image1 = self.data[t0]
            image2 = self.data[t1+1]
        return torch.FloatTensor(image1), torch.FloatTensor(image2)
