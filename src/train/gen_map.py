import time

import torch
import process.generate as gen
import dataset.factory as df
import torch.nn as nn
import h5py as h5
import os

dataset_factory = df.dataset_factory


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.model_path = "../train/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl"
args.model_name = "ResNet"
args.dataset = "CIFAR_10"
args.train_dir = "./data/train_data/"
args.batch_size = 1
args.workers = 1
args.size = (8, 8)
args.window_processor = gen.avg_processor
args.processor_name = 'avg'
args.gen_method = gen.gen_sensitive_map_rect_greed
args.gen_method_name = 'greed'
args.output_dir = "../train/maps/"
args.offset = 0
args.length = 10
args.description = 'rect'
args.output_name = ("%s_%s_%d_%d_%s_%s_%s.h5" % (args.model_name, args.dataset, args.offset, args.length,
                                              args.processor_name, args.gen_method_name, args.description))
args.use_cuda = True


def main():
    # time.sleep(36000)
    nn.Conv2d()
    model = torch.load(args.model_path).cuda()
    dataset = dataset_factory[args.dataset](args.train_dir, True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    x, y, map = gen.gen_on_set(model, loader, args.size, criterion, args.window_processor, args.gen_method,
                               args.offset, args.length, args.use_cuda)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    file = h5.File(os.path.join(args.output_dir, args.output_name))
    file.create_dataset("x", data=x.numpy())
    file.create_dataset("y", data=y.numpy())
    file.create_dataset("map", data=map.numpy())
    file.close()


if __name__ == "__main__":
    main()
