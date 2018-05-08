import h5py as h5
import torch
import numpy as np
import os

import process.apply as apply


class Arg(object):

    def __init__(self):
        super().__init__()


args = Arg()
args.map_path = '/home/jt/codes/bs/nb/src/train/maps/ResNet_CIFAR_10_0_10_avg_greed.h5'
args.apply_method = apply.apply_avg
args.apply_method_name = "avg"
args.print_frequency = 100


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
        if (i + 1) % args.print_frequency == 0:
            print('%d pics applied' % (i + 1))
    output.create_dataset("applied", data=applied.numpy())

    # print(applied[1, :, :, :])

    output.close()
    file.close()

    print('Apply completed')


if __name__ == "__main__":
    main()
