"""This module has custom samplers that are used mainly for test time inference in TDCVAE.
They are used in the jupyter notebook for TDCVAE.

"""
import numpy as np
from torch.utils.data.sampler import Sampler

class ClassSampler(Sampler):
    """This class is filtering after classes and take all images or
    a single image (repeat that up to N times)"""
    def __init__(self, dataset, picked_class, num_samples=None):
        self.indices = np.where(dataset.targets == picked_class)[0]
        num_samples = dataset.data.shape[0] if num_samples is None else num_samples
        N = len(self.indices)
        if num_samples > N:
            indices = np.random.choice(self.indices, num_samples-N)
            self.indices = np.concatenate((self.indices, indices))
        elif num_samples < N:
            self.indices = self.indices[:num_samples]
        self.indices = self.indices.tolist()

    # iterate over indices of dataset elements
    def __iter__(self):
        return iter(self.indices)

    # __len__ method that returns the length of the returned iterators.
    def __len__(self):
        return len(self.indices)

class SingleDataPointSampler(Sampler):
    """Samples a single example and repeats up to N (data points) times"""
    def __init__(self, dataset, num_samples=None):
        N = dataset.data.shape[0] if num_samples is None else num_samples
        idx = np.random.choice(N, 1)
        self.sample_repeated = np.repeat(idx, N).tolist()

    def __iter__(self):
        return iter(self.sample_repeated)

    def __len__(self):
        return len(self.sample_repeated)