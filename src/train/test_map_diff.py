import sys

sys.path.append('.')
sys.path.append('..')

import argparse
import os
import time

import torch
import torch.nn as nn
import h5py as h5

from process.apply import apply_methods
from dataset.MapValDataset import MapValDataset
import numpy as np

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-print_freq', type=int, default=100)
parser.add_argument('-classes', type=int, default=3)
parser.add_argument('-map_dir', type=str,
                    default="/home/jt/codes/bs/gp/res/maps/Deeplab_CIFAR_10_unpreprocessed_VGG16_validate.h5")
parser.add_argument('-dataset', type=str, default='CIFAR_10')
parser.add_argument('-pretrained', type=bool, default=False)
parser.add_argument('-model', type=str, default='ResNet101')
parser.add_argument('-model_path', type=str,
                    default='/home/jt/codes/bs/gp/res/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl')
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='unpreprocessed_ResNet')
parser.add_argument('-threshold', type=float, default=0.9)
parser.add_argument('-apply_method', type=str, default='apply_loss4D')
parser.add_argument('-output', type=str, default="./output")
parser.add_argument('-repeat', type=int, default=1)

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method]
args.batch_size = 1


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


def find_diff(val_loader, model, criterion, apply_method=None, threshold=1.0):

    # switch to evaluate mode
    model.eval()
    x0 = []
    y0 = []
    loss0 = []
    x1 = []
    y1 = []
    loss1 = []
    label = []
    id = []

    end = time.time()
    for i, (input, target, maps) in enumerate(val_loader):
        input = input.unsqueeze(0)
        maps = maps.unsqueeze(0)
        target_tensor = torch.zeros(1)
        target_tensor[0] = target
        for j in range(maps.size(0)):
            maps[j, :, :] = (maps[j, :, :] - torch.min(maps[j, :, :])) / (
                        torch.max(maps[j, :, :]) - torch.min(maps[j, :, :]))
        maps[maps > threshold] = 1
        maps[maps <= threshold] = 0
        applied = apply_method(input, maps)

        input_var = torch.autograd.Variable(input, volatile=True)
        applied_var = torch.autograd.Variable(applied, volatile=True)
        target_var = torch.autograd.Variable(target_tensor, volatile=True).long()
        if args.use_cuda:
            input_var = input_var.cuda()
            applied_var = applied_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        applied_output = model(applied_var)
        _, output_label = output.topk(1, 1, True, True)
        _, applied_label = applied_output.topk(1, 1, True, True)
        loss = criterion(output, target_var)
        applied_loss = criterion(applied_output, target_var)

        if applied_loss.data[0] < loss.data[0]:
            x0.append(input)
            y0.append(output_label)
            loss0.append(loss.data[0])
            x1.append(applied)
            y1.append(applied_label)
            loss1.append(applied_loss[0])
            label.append(target)
            id.append(i)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t {2} diffs found'.format(
                i, len(val_loader), len(x0)))

    return x0, y0, loss0, x1, y1, loss1, label, id


def main(threshold):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    model = torch.load(args.model_path)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    map_dataset = MapValDataset(args.map_dir)

    # p1, p5 = validate(val_loader, model, criterion)
    # print('Validate result without map: top1 {0}, top5 {1}, all {2}'.format(p1, p5, all))
    x0, y0, loss0, x1, y1, loss1, label, id = find_diff(map_dataset, model, criterion, args.apply_method, threshold)

    diff_len = len(x0)
    depth = x0[0].shape[1]
    height = x0[0].shape[2]
    width = x0[0].shape[3]
    x0_numpy = np.zeros((diff_len, depth, height, width))
    y0_numpy = np.zeros(diff_len)
    loss0_numpy = np.zeros(diff_len)
    x1_numpy = np.zeros((diff_len, depth, height, width))
    y1_numpy = np.zeros(diff_len)
    loss1_numpy = np.zeros(diff_len)
    label_numpy = np.zeros(diff_len)
    id_numpy = np.zeros(diff_len)

    for i in range(diff_len):
        x0_numpy[i, :, :, :] = x0[i][0, :, :, :]
        y0_numpy[i] = y0[i]
        loss0_numpy[i] = loss0[i]
        x1_numpy[i, :, :, :] = x1[i][0, :, :, :]
        y1_numpy[i] = y1[i]
        loss1_numpy[i] = loss1[i]
        label_numpy[i] = label[i]
        id_numpy[i] = id[i]

    file = h5.File(args.output)
    file.create_dataset('x0', data=x0_numpy)
    file.create_dataset('y0', data=y0_numpy)
    file.create_dataset('loss0', data=loss0_numpy)
    file.create_dataset('x1', data=x1_numpy)
    file.create_dataset('y1', data=y1_numpy)
    file.create_dataset('loss1', data=loss1_numpy)
    file.create_dataset('label', data=label_numpy)
    file.create_dataset('id', data=id_numpy)
    file.close()


if __name__ == '__main__':
    main(args.threshold)
