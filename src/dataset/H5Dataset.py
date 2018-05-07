import torch
import torch.utils.data as data
import os
import h5py as h5


class H5Dataset(data.Dataset):
    def __init__(self, root):
        self.root = os.path.expanduser(root)
        file = h5.File(root)
        self.data = torch.from_numpy(file['x'][:]).float()
        self.label = torch.from_numpy(file['y'][:]).long()
        self.data_size = self.data.size(0)

    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index]

    def __len__(self):
        return self.data_size