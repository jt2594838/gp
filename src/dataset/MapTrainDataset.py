import torch.utils.data as data
import os
import h5py as h5
import numpy as np


"""
This dataset uses HDF5 file as its data source to provide inputs and target for training a generator.
The HDF5 file must contain 2 attributes:
    'x'   : the input data, must be a 4d array and each dimension means [number, depth, height, width] respectively
    'map' : the process map of the data, must be a 3d array. y[i, :] is the process map of x[i, :].
Data and maps are stored as numpy ndarraies.
"""


class MapTrainDataset(data.Dataset):
    def __init__(self, root, preprocess=True, threshold=0, **kwargs):
        """

        :param root:
        :param preprocess:
            When set to true, the maps will be binarized using this threshold.
        :param threshold：
        :param kwargs:
        """
        file = h5.File(root)

        x = file['x']
        y = file['map']
        data_size = x.shape[0]
        map_size = y.shape[0]

        if data_size != map_size:
            print('Data size {} is not equal to map size {}'.format(data_size, map_size))

        self.data = np.array(x[:, :, :, :])
        self.target = np.array(y[:, :, :])
        self.data_size = data_size

        if preprocess:
            self.target[self.target > threshold] = 1.0
            self.target[self.target < threshold] = 0.0
        file.close()
        print('Dataset contains {} samples'.format(self.data_size))

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.target[index, :, :]

    def __len__(self):
        return self.data_size
