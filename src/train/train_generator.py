import sys

sys.path.append('.')
sys.path.append('..')

from nets.ConvDeconvV2 import ConvDeconvV2
from nets.ConvDeconv import ConvDeconv
from nets.deeplab import Deeplab
from dataset.MapTrainDataset import MapTrainDataset

import os
import time

import torch
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-weight_decay', type=float, default=1e-4)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-print_freq', type=int, default=1)
parser.add_argument('-classes', type=int, default=3)
parser.add_argument('-train_dir', type=str, default="/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5")
parser.add_argument('-val_dir', type=str, default="/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5")
parser.add_argument('-dataset', type=str, default='anzhen_4300_zero')
parser.add_argument('-in_channels', type=int, default=1)
parser.add_argument('-pretrained', type=bool, default=False)
parser.add_argument('-model', type=str, default='ConvDeconv')
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-model_path', type=str, default='/home/jt/codes/bs/gp/res_anzhen/model')
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='unpreprocessed_ResNet')
parser.add_argument('-preprocess', type=bool, default=False)


args = parser.parse_args()


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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * loss {loss.val:.3f} avg_loss {loss.avg:.3f}'
          .format(loss=losses))

    return losses.avg


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    if args.model == 'Deeplab':
        model = Deeplab(1, args.in_channels, args.pretrained)
    elif args.model == 'ConvDeconv':
        model = ConvDeconv(args.in_channels)
    elif args.model == 'ConvDeconvV2':
        model = ConvDeconvV2(args.in_channels)
    else:
        print('Illegal model name {0}'.format(args.model))
        exit(-1)

    train_dataset = MapTrainDataset(args.train_dir, True, args.preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.MSELoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for i in range(args.epoch):
        train(train_loader, model, criterion, optimizer, i)

    val_dataset = MapTrainDataset(args.val_dir, False, args.preprocess)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    loss = validate(val_loader, model, criterion)

    filename = '{0}_{1}_{2}_{3}_{4}_{5}.pkl'.format(args.model, args.dataset, str(args.classes), str(args.epoch),
                                                    str(loss), args.description)
    path = os.path.join(args.model_path, filename)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model, path)


if __name__ == '__main__':
    main()

