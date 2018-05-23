import sys

sys.path.append('.')
sys.path.append('..')

import argparse
import h5py as h5
import matplotlib.pylab as plt
import os

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-input', type=str)
parser.add_argument('-output', type=str)

args = parser.parse_args()


def main():
    input_file = h5.File(args.input)
    args.output = os.path.join(args.output, os.path.basename(args.input))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    x0 = input_file['x0'][:]
    y0 = input_file['y0'][:]
    loss0 = input_file['loss0'][:]
    x1 = input_file['x1'][:]
    y1 = input_file['y1'][:]
    loss1 = input_file['loss1'][:]
    label = input_file['label'][:]
    id = input_file['id'][:]

    for i in range(x0.shape[0]):
        pic = x0[i]
        pic_name = '{}_{}/{}_{}.jpg'.format(id[i], y0[i], label[i], loss0[i])
        pic_path = os.path.join(args.output, pic_name)
        plt.imsave(pic_path, pic.numpy(), cmap='gray')

        pic = x1[i]
        pic_name = '{}_{}/{}_{}_applied.jpg'.format(id[i], y1[i], label[i], loss1[i])
        pic_path = os.path.join(args.output, pic_name)
        plt.imsave(pic_path, pic.numpy(), cmap='gray')
        if (i + 1) % 100 == 0:
            print('{} pics exported'.format(i + 1))
    print('Pic export over.')


if __name__ == '__main__':
    main()