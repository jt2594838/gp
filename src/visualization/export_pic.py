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


def main(input):
    input_file = h5.File(input)
    output = os.path.join(args.output, os.path.basename(input))
    if not os.path.exists(output):
        os.mkdir(output)

    x = torch.from_numpy(input_file['x'][:])
    y = torch.from_numpy(input_file['y'][:]).long()
    map = None
    apply_method = None
    if args.use_map:
        maps = torch.from_numpy(input_file['map'][:])
        apply_method = apply_methods[args.apply_method]

    for i in range(x.shape[0]):
        pic = x[i]
        pic_name = '{}_{}.jpg'.format(i, y[i])
        pic_path = os.path.join(output, pic_name)
        if args.use_map:
            map = maps[i]
            for j in range(maps.shape[0]):
                maps[j, :, :] = (maps[j, :, :] - torch.min(maps[j, :, :])) / (
                        torch.max(maps[j, :, :]) - torch.min(maps[j, :, :]))
            applied = apply_method(pic, map).squeeze()
            map_name = '{}_{}_map.jpg'.format(i, y[i])
            map_path = os.path.join(output, map_name)
            applied_name = '{}_{}_applied.jpg'.format(i, y[i])
            applied_path = os.path.join(output, applied_name)
            map = map.squeeze()
            plt.imsave(map_path, map.numpy(), cmap='gray')
            plt.imsave(applied_path, applied.numpy(), cmap='gray')
        pic = pic.squeeze()
        plt.imsave(pic_path, pic.numpy(), cmap='gray')
        if (i + 1) % 100 == 0:
            print('{} pics exported'.format(i + 1))
    print('Pic export of {} over.'.format(input))


if __name__ == '__main__':
    if os.path.isfile(args.input):
        main(args.input)
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            file = os.path.join(args.input, file)
            if os.path.isfile(file):
                main(file)
    else:
        print('invalid path {}'.format(args.input))
