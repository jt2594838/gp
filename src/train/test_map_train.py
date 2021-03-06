import sys

sys.path.append('.')
sys.path.append('..')

import argparse
import os
import time

import torch
import torch.nn as nn

from dataset.H5Dataset import H5Dataset
import nets.nets as nets
from process.apply import apply_methods

parser = argparse.ArgumentParser(description='Use processed pics to both train and validate a classifier')
parser.add_argument('-batch_size', type=int, default=25)
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-weight_decay', type=float, default=1e-4)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-print_freq', type=int, default=1)
parser.add_argument('-classes', type=int, default=3)
parser.add_argument('-train_dir', type=str)
parser.add_argument('-val_dir', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-in_channels', type=int, default=1)
parser.add_argument('-pretrained', type=bool, default=False)
parser.add_argument('-model_name', type=str)
parser.add_argument('-model_path', type=str)
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='')
parser.add_argument('-preprocess', type=bool, default=False)
parser.add_argument('-threshold', type=float, default=0.9)
parser.add_argument('-apply_method', type=str)
parser.add_argument('-output', type=str, default="./")

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method]


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
    top1 = AverageMeter()

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
        loss = criterion(output, target_var.squeeze())

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target_var.data, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):


        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        if args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, ))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1, ))

    return top1.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    print('loading model {} '.format(args.model_name))
    model = nets.classify_net_factory[args.model_name](args.classes, args.pretrained, args.in_channels, classify=True)
    print('loading training data from {}'.format(args.train_dir))
    train_dataset = H5Dataset(args.train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for i in range(args.epoch):
        train(train_loader, model, criterion, optimizer, i)

    print('loading validating data from {}'.format(args.val_dir))
    val_dataset = H5Dataset(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    p1, loss = validate(val_loader, model, criterion, args.apply_method, args.threshold)

    file = open(args.output, 'a')
    file.write('map {0} \t threshold {1} \t precision {2} \n'.format(args.map_dir, args.threshold, p1))
    file.close()


if __name__ == '__main__':
    main()
