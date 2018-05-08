import argparse
import os

import h5py as h5
import numpy as np
import torch

from process.apply import apply_methods

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-apply_method_name', type=str, default='apply_loss4D')
parser.add_argument('-print_freq', type=int, default=100)

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method_name]


def main():
    file = h5.File(args.map_path)
    output_path = args.map_path + ".applied_" + args.apply_method_name
    if os.path.exists(output_path):
        os.remove(output_path)
    output = h5.File(args.map_path + ".applied_" + args.apply_method_name)

    x = torch.from_numpy(np.array(file['x']))
    y = torch.from_numpy(np.array(file['y']))
    map = torch.from_numpy(np.array(file['map']))
    pic_cnt = x.size(0)
    print('File contains %d pics' % pic_cnt)

    applied = torch.zeros(x.size())
    for i in range(pic_cnt):
        temp = args.apply_method(x[i, :, :, :], map[i, :, :])
        applied[i, :, :, :] = temp[:, :, :]
        if (i + 1) % args.print_freq == 0:
            print('%d pics applied' % (i + 1))
    output.create_dataset("applied", data=applied.numpy())

    # print(applied[1, :, :, :])

    output.close()
    file.close()

    print('Apply completed')


if __name__ == "__main__":
    main()
