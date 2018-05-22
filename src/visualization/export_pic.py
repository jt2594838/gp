import sys

import torch

sys.path.append('.')
sys.path.append('..')

import argparse
import h5py as h5
import matplotlib.pylab as plt
import os
from process.apply import apply_methods

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-input', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-apply_method', type=str)
parser.add_argument('-use_map', type=bool, default=False)

args = parser.parse_args()


def main():
    input_file = h5.File(args.input)
    args.output = os.path.join(args.output, os.path.basename(args.input))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    x = torch.from_numpy(input_file['x'][:])
    y = torch.from_numpy(input_file['y'][:])
    map = None
    apply_method = None
    if args.use_map:
        map = torch.from_numpy(input_file['y'][:])
        apply_method = apply_methods[args.apply_method]

    for i in range(x.shape[0]):
        pic = x[i]
        pic_name = '{}_{}.jpg'.format(i, y[i])
        pic_path = os.path.join(args.output, pic_name)
        if args.use_map:
            map = map[i]
            pic = apply_method(pic, map)
            map_name = '{}_{}_map.jpg'.format(i, y[i])
            map_path = os.path.join(args.output, map_name)
            plt.imsave(map_path, map.numpy())
        plt.imsave(pic_path, pic.numpy())
        if (i + 1) % 100 == 0:
            print('{} pics exported'.format(i + 1))
    print('Pic export over.')


if __name__ == '__main__':
    main()