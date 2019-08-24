import numpy as np
from torch.utils.data.sampler import Sampler

# filter after classes and take all images or filter after classes and take a single image (repeat that up to N times)
class ClassSampler(Sampler):
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

    # provide an __iter__ method, providing a way
    # to iterate over indices of dataset elements
    def __iter__(self):
        return iter(self.indices)

    # __len__ method that returns the length of the returned iterators.
    def __len__(self):
        return len(self.indices)

# samples a single example and repeats up to N (data points) times
class SingleDataPointSampler(Sampler):
    def __init__(self, dataset, num_samples=None):
        N = dataset.data.shape[0] if num_samples is None else num_samples
        idx = np.random.choice(N, 1)
        self.sample_repeated = np.repeat(idx, N).tolist()

    def __iter__(self):
        return iter(self.sample_repeated)
    
    def __len__(self):
        return len(self.sample_repeated)