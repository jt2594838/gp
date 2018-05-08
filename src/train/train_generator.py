import os
import time

import torch
import torch.nn as nn

from nets.ConvDeconv import ConvDeconv
from nets.deeplab import Deeplab
from dataset.MapTrainDataset import MapTrainDataset


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.batch_size = 10
args.workers = 1
args.lr = 0.005
args.weight_decay = 1e-4
args.epoch = 50
args.print_freq = 1
args.classes = 1
args.train_dir = "/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5"
args.val_dir = "/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5"
args.dataset = 'anzhen_4300_zero'
args.momentum = 0.9
args.model = 'Deeplab'
args.model_path = '/home/jt/codes/bs/gp/res_anzhen/generator_model'
args.description = 'unpreprocessed_ResNet'
args.preprocess = False
args.usecuda = True
args.in_channels = 1
args.pretrained = False
args.gpu_no = "0"


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
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if args.usecuda:
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
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

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

    print(' * Prec@1 {loss.val:.3f} Prec@5 {loss.avg:.3f}'
          .format(loss=losses))

    return losses.avg


def main():
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    if args.model == 'Deeplab':
        model = Deeplab(1, args.in_channels, args.pretrained)
    elif args.model == 'ConvDeconv':
        model = ConvDeconv(args.in_channels)
    else:
        print('Illegal model name {0}'.format(args.model))
        exit(-1)

    train_dataset = MapTrainDataset(args.train_dir, True, args.preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.MSELoss()

    if args.usecuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              weight_decay=args.weight_decay)

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

