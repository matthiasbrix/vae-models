from torch.utils.data import Dataset
from sklearn.datasets import fetch_lfw_people
import scipy.io
import torch
import numpy as np
import skimage
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
    def __init__(self, volumes, resize=None, crop=None, sampling=False):
        asd = load_all_volumes(volumes)
        a = [v[:40] for v in asd] # take the first 40 because usually the last ~25 are not so informative
        for i in range(len(a)): # normalize to (0,1)
            norm = (a[i] - np.min(a[i]))/(np.max(a[i]) - np.min(a[i]))
            a[i] = norm
        self.data = np.concatenate(tuple(a), axis=0) # shape is 192 x 384 x 384, as 192=64*3
        self.transform = skimage.transform.resize if resize else None
        self.resized_dims = resize
        self.crop = crop
        self.sampling = sampling
        self.timestamps = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, t0):
        t1 = (t0+1)
        t1 = t1 if t1 < self.data.shape[0] else (t0-1)
        if self.sampling:
            self.timestamps.append(t0)
        if self.transform is not None and self.resized_dims is not None:
            image1 = self.transform(self.data[t0], output_shape=self.resized_dims, anti_aliasing=True)
            image2 = self.transform(self.data[t1], output_shape=self.resized_dims, anti_aliasing=True)
        if self.crop:
            if self.crop[0]:
                image1 = image1[self.crop[0][0]:self.crop[0][1], :]
                image2 = image2[self.crop[0][0]:self.crop[0][1], :]
            if self.crop[1]:
                image1 = image1[:, self.crop[1][0]:self.crop[1][1]]
                image2 = image2[:, self.crop[1][0]:self.crop[1][1]]
        if not self.transform and not self.crop:
            image1 = self.data[t0]
            image2 = self.data[t1]
        image1 = np.expand_dims(image1, axis=0) # inserts a channel dim at 0th index
        image2 = np.expand_dims(image2, axis=0) # inserts a channel dim at 0th index
        return torch.FloatTensor(image1), torch.FloatTensor(image2)
