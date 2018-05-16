import torch
import torch.utils.data as data
import os
import h5py as h5


class MapValDataset(data.Dataset):
    def __init__(self, root):
        self.root = os.path.expanduser(root)
        file = h5.File(root)
        self.x = torch.from_numpy(file['x'][:])
        self.y = torch.from_numpy(file['y'][:])
        self.map = torch.from_numpy(file['map'][:])
        self.data_size = self.x.size(0)
        file.close()

    def __getitem__(self, index):
        return self.x[index, :, :, :], self.y[index], self.map[index, :, :]

    def __len__(self):
        return self.data_size
