import torch
import torch.utils.data as data
import os
import h5py as h5
from torchvision.transforms import transforms
import numpy as np


class H5Dataset(data.Dataset):
    def __init__(self, root, train=True, sample_rate=0.9):
        self.root = os.path.expanduser(root)
        file = h5.File(root)
        data_size = file['x'].shape[0]
        if train:
            choices = np.random.choice(data_size, int(sample_rate * data_size), replace=False)
        else:
            choices = np.random.choice(data_size, int((1 - sample_rate) * data_size), replace=False)
        choices.sort()
        self.data = torch.from_numpy(file['x'][list(choices)]).float().unsqueeze(1)
        self.label = torch.from_numpy(file['y'][list(choices)]).long()
        self.data_size = len(choices)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        print('Distribution: {0}, {1}, {2}'.format(self.label[self.label == 0].size(0),
                                                   self.label[self.label == 1].size(0),
                                                   self.label[self.label == 2].size(0),))

    def __getitem__(self, index):
        data = self.data[index, :, :]
        return self.transform(data), self.label[index, 0]

    def __len__(self):
        return self.data_size


def get_anzhen_dataset(root, train, sample_rate=0.9):
    return H5Dataset(root, train, sample_rate=sample_rate)