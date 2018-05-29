import sys
sys.path.append('.')
sys.path.append('..')

import argparse

import h5py as h5
import numpy as np
from dataset.factory import dataset_factory


parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-input', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-train', type=bool, default=False)
parser.add_argument('-output', type=str)

args = parser.parse_args()


def main():
    dataset = dataset_factory[args.dataset](args.input, args.train)
    x = None
    y = None
    for i, (input, target) in enumerate(dataset):
        if x is None:
            x = np.zeros((len(dataset), input.shape[0], input.shape[1], input.shape[2]))
            y = np.zeros((len(dataset)))
        x[i, :, :, :] = input[:]
        y[i] = target

    output = h5.File(args.output)
    output.create_dataset('x', data=x)
    output.create_dataset('y', data=y)
    output.close()


if __name__ == '__main__':
    main()
