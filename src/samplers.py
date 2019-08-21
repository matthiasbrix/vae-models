from torch.utils.data.sampler import Sampler
import numpy as np

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
