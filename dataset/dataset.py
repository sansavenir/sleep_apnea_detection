from __future__ import print_function, division
import sys
sys.path.append('../')
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from dataset.parser import parse_file
import numpy as np

class Dataset(Dataset):

    def __init__(self, names, is_training=True, window_size=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data, self.target = [],[]
        for name in names:
          a, b = parse_file(name)
          self.data.append(a)
          self.target.append(b)
        self.data = np.concatenate(self.data)
        self.target = np.concatenate(self.target)

        ones = np.count_nonzero(self.target)
        self.weights = ones / (self.target.shape[0] - ones)

        self.window_size = window_size
        self.x, self.y = [], []
        for i in range((self.data.shape[0] - 1) // self.window_size):
          if is_training:
            if self.target[(i + 1) * self.window_size] == 1 or np.random.uniform(0,1) <= self.weights:
              self.x.append(self.data[i * self.window_size:(i + 1) * self.window_size])
              self.y.append((self.target[(i + 1) * self.window_size]))
          else:
            self.x.append(self.data[i * self.window_size:(i + 1) * self.window_size])
            self.y.append((self.target[(i + 1) * self.window_size]))
        self.x = Variable(torch.Tensor(self.x).float(), requires_grad=False)
        self.y = Variable(torch.Tensor(self.y).float(), requires_grad=False)

        assert self.x.shape[0] == self.y.shape[0]

    def _one_hot(self,x):
      if x == 0.:
        return [1,0]
      else:
        return [0,1]
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]