import torch
import torch.utils.data as data
import os
import h5py as h5


"""
This dataset uses HDF5 file as its data source to provide inputs and target for testing a generator.
The HDF5 file must contain 2 attributes:
    'x'   : the input data, must be a  4d array and each dimension means [number, depth, height, width] respectively
    'y' : the label of the data, must be a 1d array. y[i] is the label of x[i, :].
    'map' : the process map of the data, must be a 3d array. y[i, :] is the process map of x[i, :].
Data, labels and maps are stored as numpy ndarraies.
"""


class MapValDataset(data.Dataset):
    def __init__(self, root):
        file = h5.File(root)

        data_size = file['x'].shape[0]
        label_size = file['y'].shape[0]
        map_size = file['map'].shape[0]

        if data_size != label_size or data_size != map_size or label_size != map_size:
            print("Inconsistent sizes, data size {}, label size {}, map size {}".format(data_size, label_size, map_size))

        self.x = torch.from_numpy(file['x'][:])
        self.y = torch.from_numpy(file['y'][:]).long()
        self.map = torch.from_numpy(file['map'][:])
        self.data_size = data_size
        file.close()

    def __getitem__(self, index):
        return self.x[index, :, :, :], self.y[index], self.map[index, :, :]

    def __len__(self):
        return self.data_size
