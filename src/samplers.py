import torch
from torch.utils.data.sampler import Sampler
import numpy as np

# filter after classes and take all images or filter after classes and take a number of images (likely just one)
class ClassSampler(Sampler):
    def __init__(self, dataset, picked_class, num_samples=None):
        self.mask = picked_class
        self.indices = np.where(dataset.targets == self.mask)[0] # 10, 20, 30, 40, ..., 100 # size=10
        if num_samples is not None:
            filter_indices = torch.randint(0, len(self.indices), (num_samples,)) # 5 from self.indices, say, 0...4 so indices for numbers 0, 10, ..., 40
            self.indices = [self.indices[i] for i in filter_indices]
        torch.tensor(self.indices)

    # provide an __iter__ method, providing a way
    # to iterate over indices of dataset elements
    def __iter__(self):
        return iter(self.indices)

    # __len__ method that returns the length of the returned iterators.
    def __len__(self):
        return len(self.indices)
