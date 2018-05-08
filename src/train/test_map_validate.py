import argparse
import os
import time

import torch
import torch.nn as nn

from dataset.factory import dataset_factory
from process.apply import apply_methods
from dataset.MapValDataset import MapValDataset


parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-weight_decay', type=float, default=1e-4)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-print_freq', type=int, default=1)
parser.add_argument('-classes', type=int, default=3)
parser.add_argument('-map_dir', type=str, default="/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_unpreprocessed_0.09455910949409008_VGG16_0.9_79.11_98.59_validate.h5")
parser.add_argument('-val_dir', type=str, default="./data/val_data/")
parser.add_argument('-dataset', type=str, default='CIFAR_10')
parser.add_argument('-in_channels', type=int, default=1)
parser.add_argument('-pretrained', type=bool, default=False)
parser.add_argument('-model', type=str, default='ResNet101')
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-model_path', type=str, default='/home/jt/codes/bs/nb/src/train/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl')
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='unpreprocessed_ResNet')
parser.add_argument('-preprocess', type=bool, default=False)
parser.add_argument('-threshold', type=float, default=0.9)
parser.add_argument('-apply_method', type=str, default='apply_loss4D')

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


def validate(val_loader, model, criterion, map_dataset=None, apply_method=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_size = val_loader.batch_size

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if map_dataset is not None:
            maps = map_dataset[i * batch_size: i * batch_size + input.size(0)]
            for j in range(maps.size(0)):
                maps[j, :, :] = (maps[j, :, :] - torch.min(maps[j, :, :])) / (torch.max(maps[j, :, :]) - torch.min(maps[j, :, :]))
            threshold = args.threshold
            maps[maps > threshold] = 1
            maps[maps <= threshold] = 0
            input = apply_method(input, maps)

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
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5,))

    return top1.avg, top5.avg


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
    if not os.path.exists(args.val_dir):
        os.makedirs(args.val_dir)

    model = torch.load(args.model_path)
    val_dataset = dataset_factory[args.dataset](args.val_dir, False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
        model = model.cuda()
        criterion = criterion.cuda()

    map_dataset = MapValDataset(args.map_dir)

    # p1, p5 = validate(val_loader, model, criterion)
    # print('Validate result without map: top1 {0}, top5 {1}, all {2}'.format(p1, p5, all))
    p1, p5 = validate(val_loader, model, criterion, map_dataset, args.apply_method)
    print('Validate result with map: top1 {0}, top5 {1}'.format(p1, p5))


if __name__ == '__main__':
    main()

