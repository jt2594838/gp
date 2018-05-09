import torch.utils.data as data
import os
import h5py as h5
import numpy as np


class MapTrainDataset(data.Dataset):
    def __init__(self, root, train=True, preprocess=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        file = h5.File(root)
        x = file['x']
        y = file['map']
        data_size = x.shape[0]
        if train:
            self.data = np.array(x[:int(0.9 * data_size), :, :, :])
            self.target = np.array(y[:int(0.9 * data_size), :, :])
            self.data_size = self.data.shape[0]
        else:
            self.data = np.array(x[int(0.9 * data_size):, :, :, :])
            self.target = np.array(y[int(0.9 * data_size):, :, :])
            self.data_size = self.data.shape[0]
        if preprocess:
            self.target[self.target > 0.5] = 1.0
            self.target[self.target <= 0.5] = 0.0

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.target[index, :, :]

    def __len__(self):
        return self.data_size
