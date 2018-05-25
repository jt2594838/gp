import sys

sys.path.append('.')
sys.path.append('..')

import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np

from process.apply import apply_methods
from dataset.MapValDataset import MapValDataset
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-batch_size', type=int, default=50)
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
parser.add_argument('-threshold', type=str, default="0.9, 1.0")
parser.add_argument('-apply_method', type=str, default='apply_loss4D')
parser.add_argument('-output', type=str, default="./output")
parser.add_argument('-repeat', type=int, default=10)
parser.add_argument('-criterion', type=str)

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


def validate(val_loader, model, criterion, apply_method=None, threshold=1.0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion_matrix = torch.zeros((args.classes, args.classes))
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, maps) in enumerate(val_loader):
        for j in range(maps.size(0)):
            maps[j, :, :] = (maps[j, :, :] - torch.min(maps[j, :, :])) / (
                        torch.max(maps[j, :, :]) - torch.min(maps[j, :, :]))
        maps[maps > threshold] = 1
        maps[maps <= threshold] = 0
        input = apply_method(input, maps)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        if args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,), confusion_matrix=confusion_matrix)
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
                top1=top1,))

    all_prec = 0
    for i in range(args.classes):
        all_prec += confusion_matrix[i, i]
        all_prec /= torch.sum(confusion_matrix)

    all_recall = 0
    for i in range(1, args.classes):
        all_recall += confusion_matrix[i, i]
        all_recall /= torch.sum(confusion_matrix[1:, :])

    precs = []
    recalls = []
    for i in range(args.classes):
        prec = confusion_matrix[i, i] / torch.sum(confusion_matrix[:, i])
        recall = confusion_matrix[i, i] / torch.sum(confusion_matrix[i, :])
        precs.append(prec)
        recalls.append(recall)

    print(' * All prec {}, All recall {}, precs {}, recalls {}, loss {}'
          .format(all_prec, all_recall, precs, recalls, losses.avg))

    return all_prec, all_recall, precs, recalls, losses.avg


def validate_auc(val_loader, model, apply_method=None, threshold=1.0):
    batch_time = AverageMeter()
    labels = None
    scores = None

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, maps) in enumerate(val_loader):
        if labels is None:
            labels = np.zeros(len(val_loader))
            scores = np.zeros(len(val_loader))

        for j in range(maps.size(0)):
            maps[j, :, :] = (maps[j, :, :] - torch.min(maps[j, :, :])) / (
                    torch.max(maps[j, :, :]) - torch.min(maps[j, :, :]))
        # maps[maps > threshold] = 1
        maps[maps <= threshold] = 0
        input = apply_method(input, maps)

        input_var = torch.autograd.Variable(input, volatile=True)
        if args.use_cuda:
            input_var = input_var.cuda()

        # compute output
        sigmoid = nn.Sigmoid()
        output = sigmoid(model(input_var))
        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        output = output / torch.sum(output)

        labels[i] = 0 if target[0] == 0 else 1
        scores[i] = 1.0 - output.data[0, 0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time))

    auc_roc = roc_auc_score(labels, scores)
    print(' * auc_roc {}'
          .format(auc_roc))

    return auc_roc


def accuracy(output, target, topk=(1,), confusion_matrix=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_correct = pred.eq(target.view(1, -1).expand_as(pred))
    if confusion_matrix is not None:
        for i in range(pred.size(1)):
            confusion_matrix[target[i]][pred[0, i]] += 1

    res = []
    for k in topk:
        correct_k = pred_correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main(threshold, map_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    model = torch.load(args.model_path)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    map_dataset = MapValDataset(map_dir)
    val_loader = torch.utils.data.DataLoader(
        map_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.criterion == 'prec':
        all_prec, all_recall, precs, recalls, loss = validate(val_loader, model, criterion, args.apply_method, threshold)
        print('Validate result with map: {0} {1} {2} {3} {4} {5}, threshold {2}'.format(args.criterion, all_prec, all_recall, precs, recalls, loss, threshold))
        file_path = args.output
        if args.use_dir:
            file_path = os.path.join(file_path, os.path.basename(map_dir))
        file = open(file_path, 'a')
        file.write('map {0} \t threshold {1} \t all_prec {2} all_recall {3} precs {4} recalls {5} loss {6}\n'.format(
            args.map_dir, threshold,  all_prec, all_recall, precs, recalls, loss))
        file.close()
    elif args.criterion == 'auc_roc':
        auc_roc = validate_auc(val_loader, model, args.apply_method, threshold)
        print('Validate result with map: auc_roc {0}, threshold {1}'.format(auc_roc, threshold))
        file_path = args.output
        if args.use_dir:
            file_path = os.path.join(file_path, os.path.basename(map_dir))
        file = open(file_path, 'a')
        file.write('map {0} \t threshold {1} \t auc_roc {2} \n'.format(args.map_dir, threshold, auc_roc))
        file.close()
    else:
        print('Invalid criterion {}'.format(args.criterion))
        exit(-1)


if __name__ == '__main__':
    args.threshold = args.threshold.split(',')
    args.use_dir = False
    if os.path.isfile(args.map_dir):
        for threshold in args.threshold:
            main(float(threshold), args.map_dir)
    elif os.path.isdir(args.map_dir):
        args.use_dir = True
        for file in os.listdir(args.map_dir):
            file = os.path.join(args.map_dir, file)
            if os.path.isfile(file):
                for threshold in args.threshold:
                    main(float(threshold), file)
    else:
        print('invalid path {}'.format(args.input))




