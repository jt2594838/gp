import os
import time

import torch
import torch.nn as nn

import nets.nets as nets
from dataset.factory import dataset_factory


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.batch_size = 50
args.workers = 1
args.lr = 0.01
args.weight_decay = 1e-4
args.epoch = 200
args.print_freq = 1
args.classes = 3
args.train_dir = "/home/jt/codes/bs/gp/data/anzhen/merged2"
args.val_dir = "/home/jt/codes/bs/gp/data/anzhen/merged2"
args.dataset = 'anzhen'
args.in_channels = 1
args.pretrained = False
args.model = 'ResNet101'
args.momentum = 0.9
args.model_path = '/home/jt/codes/bs/gp/res_anzhen/model'
args.use_cuda = True



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
    confusion_matrix = torch.zeros((args.classes, args.classes))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).squeeze()
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
            print(confusion_matrix)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion_matrix = torch.zeros((args.classes, args.classes))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()
        if args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1,  = accuracy(output.data, target, topk=(1, ), confusion_matrix=confusion_matrix)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            print(confusion_matrix)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    print(confusion_matrix)

    return top1.avg


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


def main():
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    model = nets.net_factory[args.model](args.classes, args.pretrained, args.in_channels, classify=True)
    train_dataset = dataset_factory[args.dataset](args.train_dir, True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for i in range(args.epoch):
        train(train_loader, model, criterion, optimizer, i)

    val_dataset = dataset_factory[args.dataset](args.val_dir, False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    p1 = validate(val_loader, model, criterion)

    filename = '{0}_{1}_{2}_{3}_{4}.pkl'.format(args.model, args.dataset, str(args.classes), str(args.epoch),
                                                    str(p1))
    path = os.path.join(args.model_path, filename)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model, path)


if __name__ == '__main__':
    main()


