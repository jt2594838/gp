import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os
import time

import torch
import h5py as h5

from dataset.factory import dataset_factory


parser = argparse.ArgumentParser(description='Generate process maps of a dataset using a generator')
parser.add_argument('-print_freq', type=int, default=100)
parser.add_argument('-classes', type=int, default=1)
parser.add_argument('-data_dir', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-model_name', type=str)
parser.add_argument('-model_path', type=str)
parser.add_argument('-map_path', type=str)
parser.add_argument('-limit', type=int, default=50000)
parser.add_argument('-description', type=str, default='')
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')

args = parser.parse_args()
args.description += '_l' + str(args.limit)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate(data_set, model, limit):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    map = None
    x = None
    y = None
    limit = limit if limit < len(data_set) else len(data_set)
    for i, (input, target) in enumerate(data_set):
        if input.dim() == 2:
            input = input.unsqueeze(0)
        elif input.dim() !=3:
            print('Illegal dimension of input : {}, which must be 2 or 3'.format(input.dim()))

        input_var = torch.autograd.Variable(input).unsqueeze(0)
        if args.use_cuda:
            input_var = input_var.cuda()

        # compute output
        output = model(input_var)

        # record inputs and outputs
        if map is None:
            map = torch.zeros((len(data_set), output.size(2), output.size(3)))
            x = torch.zeros(len(data_set), input.size(0), input.size(1), input.size(2))
            y = torch.zeros(len(data_set))
        map[i, :, :] = output.data[0, 0, :, :]
        x[i, :, :, :] = input[:, :, :]
        y[i] = target

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Generating: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i + 1, limit, batch_time=batch_time))
        if i + 1 >= limit:
            break

    print("Process map generation with generator is done")
    return map, x, y


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    model = torch.load(args.model_path)
    if args.use_cuda:
        model = model.cuda()
    train_dataset = dataset_factory[args.dataset](args.data_dir)

    map, x, y = generate(train_dataset, model, args.limit)

    filename = '{0}_{1}_{2}.h5'.format(args.model_name, args.dataset, args.description)
    path = os.path.join(args.map_path, filename)
    if not os.path.exists(args.map_path):
        os.makedirs(args.map_path)

    file = h5.File(path)
    file.create_dataset('map', data=map)
    file.create_dataset('x', data=x)
    file.create_dataset('y', data=y)
    file.close()


if __name__ == '__main__':
    main()


