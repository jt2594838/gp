import sys
sys.path.append('.')
sys.path.append('..')

import argparse

import h5py as h5
import numpy as np
import torch

from dataset.factory import dataset_factory
from process.apply import apply_methods


"""
Apply the process maps to a dataset and generate the processed pics, which can be used to test the improvement or 
iteratively train the nets.
"""


parser = argparse.ArgumentParser(description='Apply the process maps to a dataset and generate the processed pics.')
parser.add_argument('-dataset', type=str)
parser.add_argument('-dataset_dir', type=str)
parser.add_argument('-map_dir', type=str)
parser.add_argument('-apply_method_name', type=str)
parser.add_argument('-train', type=bool, default=True)
parser.add_argument('-limit', type=int)
parser.add_argument('-print_freq', type=int, default=100)

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method_name]


def apply(dataset, map, method):
    applied = None
    y = None
    limit = args.limit if args.limit <= len(dataset) else len(dataset)

    for i, (input, target) in enumerate(dataset):
        if input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim() != 4:
            print('The dimension of input is illegal {}, which must be 3 or 4'.format(input.dim()))
            exit(-1)

        if applied is None:
            applied = np.zeros((limit, input.shape[1], input.shape[2], input.shape[3]))
            y = np.zeros((limit, 1))
        applied[i, :, :, :] = method(input, torch.from_numpy(map[i, :, :]).unsqueeze(0))
        y[i, :] = target
        if (i + 1) % args.print_freq == 0:
            print('{0}/{1} applied'.format(i + 1, args.limit))
        if i + 1 >= args.limit:
            break
    return applied, y


def main():
    dataset = dataset_factory[args.dataset](args.dataset_dir, train=args.train)
    file = h5.File(args.map_dir)
    map = file['map']
    applied, y = apply(dataset, map, args.apply_method)
    file.close()

    filename = args.map_dir + '.applied'
    file = h5.File(filename)
    file.create_dataset('x', data=applied)
    file.create_dataset('y', data=y)
    file.close()


if __name__ == "__main__":
    main()
