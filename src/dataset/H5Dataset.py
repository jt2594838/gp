import torch
import torch.utils.data as data
import os
import h5py as h5
from torchvision.transforms import transforms
import numpy as np

"""
This dataset uses HDF5 file as its data source to provide inputs and target for a classifier.
The HDF5 file must contain 2 attributes:
    'x' : the input data, must be a 3d or 4d array and each dimension means [number, (depth), height, width] respectively
    'y' : the label of the data, must be a 1d array. y[i] is the label of x[i, :].
Data and label are stored as PyTorch tensors.
"""


class H5Dataset(data.Dataset):
    def __init__(self, root, num_classes=3, **kwargs):
        file = h5.File(root)
        data_size = file['x'].shape[0]
        label_size = file['y'].shape[0]

        if data_size != label_size:
            print('Inconsistent data size ({}) and label size ({}), please check your input file.'.format(data_size, label_size))
            exit(-1)
        print('File contains {0} samples'.format(self.data_size))

        self.data = torch.from_numpy(file['x'][:]).float()
        if self.data.dim() == 3:
            self.data = self.data.unsqueeze(1)
        elif self.data.dim() != 4:
            print('Invalid data dimension {}, which must be 3 or 4!'.format(self.data.dim()))
            exit(-1)
        self.label = torch.from_numpy(file['y'][:]).long()
        if self.label.dim() != 1:
            print('Labels must an 1d array, but got a {}d array'.format(self.label.dim()))
            exit(-1)
        self.data_size = data_size

        self.num_classes = num_classes
        distribution = []
        for i in range(self.num_classes):
            distribution.append(self.label[self.label == i].size(0))
        print('Sample Distribution: {0}'.format(distribution))
        file.close()

    def __getitem__(self, index):
        data = self.data[index, :]
        return data, self.label[index]

    def __len__(self):
        return self.data_size


def get_H5_dataset(root, **kwargs):
    return H5Dataset(root, **kwargs)
