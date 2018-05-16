import argparse

import h5py as h5
import numpy as np

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-input', type=str)
parser.add_argument('-sample_rate', type=float)
parser.add_argument('-output', type=str)

args = parser.parse_args()


def main():
    input = h5.File(args.input)
    data_size = input['x'].shape[0]
    choices = np.random.choice(data_size, int(args.sample_rate * data_size), replace=False)

    choices.sort()
    x = input['x'][list(choices)].float()
    y = input['y'][list(choices)].long()

    output = h5.File(args.output)
    output.create_dataset('x', data=x)
    output.create_dataset('y', data=y)

    input.close()
    output.close()


if __name__ == '__main__':
    main()