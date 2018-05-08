import os
import time

import torch
import h5py as h5

from dataset.factory import dataset_factory


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.print_freq = 100
args.classes = 1
args.data_dir = "./data/train_data/"
args.dataset = 'CIFAR_10'
args.model = 'DeeplabS'
args.model_path = '/home/jt/codes/bs/nb/src/train/models/deeplabS2048_CIFAR_5000_zero_1_10_0.09455910949409008_unpreprocessed_VGG16.pkl'
args.map_path = '/home/jt/codes/bs/nb/src/train/maps'
args.train = True
args.limit = 5000
args.description = '0.09455910949409008_unpreprocessed_VGG16'
args.description += '_train' if args.train else '_validate'
args.description += '_l' + str(args.limit)
args.use_cuda = True
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


def generate(data_set, model, limit):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    map = None
    for i, (input, target) in enumerate(data_set):
        input_var = torch.autograd.Variable(input, volatile=True).unsqueeze(0)
        if args.use_cuda:
            input_var = input_var.cuda()

        # compute output
        output = model(input_var)

        # record inputs and outputs
        if map is None:
            map = torch.zeros((len(data_set), output.size(2), output.size(3)))
        map[i, :, :] = output.data[0, 0, :, :]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Generating: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, limit, batch_time=batch_time))
        if i + 1 >= limit:
            break

    return map


def main():
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    model = torch.load(args.model_path)
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
        model = model.cuda()
    train_dataset = dataset_factory[args.dataset](args.data_dir, args.train)

    map = generate(train_dataset, model, args.limit)

    filename = '{0}_{1}_{2}.h5'.format(args.model, args.dataset, args.description)
    path = os.path.join(args.map_path, filename)
    if not os.path.exists(args.map_path):
        os.makedirs(args.map_path)

    file = h5.File(path)
    file.create_dataset('map', data=map)
    file.close()


if __name__ == '__main__':
    main()


