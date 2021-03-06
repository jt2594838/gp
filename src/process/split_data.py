import argparse

import h5py as h5
import numpy as np

"""
Split an HDF5 dataset into 3 parts: train set, validation set and test set.
The size of train set and validation set will be determined by train_rate and val_rate parameters, and the rest will be
the test set.
"""

parser = argparse.ArgumentParser(description='Split an HDF5 dataset into 3 parts: train, validation and test.')
parser.add_argument('-input', type=str)
parser.add_argument('-train_rate', type=float)
parser.add_argument('-val_rate', type=float)
parser.add_argument('-output', type=str)
parser.add_argument('-classes', type=int)

args = parser.parse_args()


def main():
    file = h5.File(args.input)

    x = file['x'][:]
    y = file['y'][:].astype(int)

    if len(x.shape) == 3:
        x = np.expand_dims(x, 1)
    elif len(x.shape) != 4:
        print('Incorrect dimension of X : {}, which must be 3 or 4'.format(len(x.shape)))
        exit(-1)

    # extract ids of different classes
    id_map = {}
    for i in range(args.classes):
        id_map[str(i)] = []

    tot_cnt = y.shape[0]

    for i in range(tot_cnt):
        id_map[str(y[i, 0])].append(i)

    # split the samples in each class according to the rates
    depth = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    train_id = []
    val_id = []
    test_id = []

    for i in range(args.classes):
        id_list = id_map[str(i)]
        class_cnt = len(id_list)

        train_cnt = int(class_cnt * args.train_rate)
        val_cnt = int(class_cnt * args.val_rate)

        for j in range(0, train_cnt):
            train_id.append(id_list[j])
        for j in range(train_cnt, train_cnt + val_cnt):
            val_id.append(id_list[j])
        for j in range(train_cnt + val_cnt, class_cnt):
            test_id.append(id_list[j])

    print('{} train samples {} val samples {} test samples'.format(len(train_id), len(val_id), len(test_id)))

    # concat the samples
    train_x = np.zeros((len(train_id), depth, height, width))
    val_x = np.zeros((len(val_id), depth, height, width))
    test_x = np.zeros((len(test_id), depth, height, width))
    train_y = np.zeros((len(train_id)))
    val_y = np.zeros((len(val_id)))
    test_y = np.zeros((len(test_id)))

    for i in range(len(train_id)):
        train_x[i, :, :, :] = x[train_id[i], :, :, :]
        train_y[i] = y[train_id[i]]

    for i in range(len(val_id)):
        val_x[i, :, :, :] = x[val_id[i], :, :, :]
        val_y[i] = y[val_id[i]]

    for i in range(len(test_id)):
        test_x[i, :, :, :] = x[test_id[i], :, :, :]
        test_y[i] = y[test_id[i]]

    # write out in separate files
    train_file = h5.File(args.output + '.train')
    train_file.create_dataset('x', data=train_x)
    train_file.create_dataset('y', data=train_y)
    train_file.close()

    val_file = h5.File(args.output + '.val')
    val_file.create_dataset('x', data=val_x)
    val_file.create_dataset('y', data=val_y)
    val_file.close()

    test_file = h5.File(args.output + '.test')
    test_file.create_dataset('x', data=test_x)
    test_file.create_dataset('y', data=test_y)
    test_file.close()


if __name__ == '__main__':
    main()
