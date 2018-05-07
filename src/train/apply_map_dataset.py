import h5py as h5
import torch
import numpy as np
import os

import process.apply as apply
from dataset.factory import dataset_factory

class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.dataset = 'CIFAR_10'
args.dataset_dir = '/home/jt/codes/bs/nb/src/train/data/train_data'
args.map_dir = '/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_0.09455910949409008_unpreprocessed_VGG16_train_l10000.h5'
args.train = True
args.apply_method = apply.apply_loss
args.limit = 5000
args.print_frequency = 100


def apply(dataset, map, method):
    applied = None
    y = None
    for i, (input, target) in enumerate(dataset):
        if applied is None:
            applied = np.zeros((args.limit, input.shape[0], input.shape[1], input.shape[2]))
            y = np.zeros((len(dataset), 1))
        applied[i, :, :, :] = method(input, torch.from_numpy(map[i, :, :]))
        y[i, :] = target
        if (i + 1) % args.print_frequency == 0:
            print('{0}/{1} applied'.format(i + 1, args.limit))
        if i + 1 >= args.limit:
            break
    return applied, y


def main():
    dataset = dataset_factory[args.dataset](args.dataset_dir, args.train)
    file = h5.File(args.map_dir)
    map = file['map']
    applied, y = apply(dataset, map, args.apply_method)
    map = None
    file.close()

    filename = args.map_dir + '.applied'
    file = h5.File(filename)
    file.create_dataset('x', data=applied)
    file.create_dataset('y', data=y)
    file.close()


if __name__ == "__main__":
    main()
