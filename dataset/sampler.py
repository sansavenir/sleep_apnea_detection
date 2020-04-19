import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset):
        self.labels_list = []
        loader = DataLoader(dataset)
        (_, self.labels) = zip(*loader)
        self.labels = list(self.labels)
        self.weights = dataset.get_weights()
        self.indexes = np.random.permutation(list(enumerate(self.labels)))

    def __iter__(self):
        for (ind,lab) in self.indexes:
            if lab == 1. or np.random.uniform(0,1) <= self.weights[1] / self.weights[0]:
                yield int(ind)

    def __len__(self):
        return len(self.labels)