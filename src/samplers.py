import torch
from torch.utils.data.sampler import Sampler
import numpy as np

# filter after classes and take all images or filter after classes and take a single image (repeat that up to N times)
class ClassSampler(Sampler):
    def __init__(self, dataset, picked_class, single_sample=False):
        self.indices = np.where(dataset.targets == picked_class)[0]
        if single_sample:
            # pick random index from self.indices and and repeat
            N = dataset.data.shape[0]
            idx = np.random.choice(self.indices, 1)
            idx_repeated = np.repeat(idx, N).tolist()
            self.indices = [i for i in idx_repeated]
        self.indices = torch.tensor(self.indices)

    # provide an __iter__ method, providing a way
    # to iterate over indices of dataset elements
    def __iter__(self):
        return iter(self.indices)

    # __len__ method that returns the length of the returned iterators.
    def __len__(self):
        return len(self.indices)

# samples a single example and repeats up to N (data points) times
class SingleDataPointSampler(Sampler):
    def __init__(self, dataset):
        N = dataset.data.shape[0]
        idx = np.random.choice(N, 1)
        self.sample_repeated = np.repeat(idx, N).tolist()

    def __iter__(self):
        return iter(self.sample_repeated)
    
    def __len__(self):
        return len(self.sample_repeated)