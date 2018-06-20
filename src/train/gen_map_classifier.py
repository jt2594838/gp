import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os

import h5py as h5
import torch
import torch.nn as nn

from dataset.factory import dataset_factory
from process.apply import apply_methods
from process.generate import gen_methods, processors, gen_on_set

parser = argparse.ArgumentParser(description='Generate process maps of a dataset using a classifier')
parser.add_argument('-data_dir', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-model_path', type=str)
parser.add_argument('-model_name', type=str)
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='')
parser.add_argument('-apply_method_name', type=str)
parser.add_argument('-size', type=int, default=8)
parser.add_argument('-gen_method_name', type=str)
parser.add_argument('-output_dir', type=str)
parser.add_argument('-offset', type=int, default=0)
parser.add_argument('-length', type=int, default=4300)
parser.add_argument('-update_err', type=bool, default=True)

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method_name]
args.size = (args.size, args.size)
args.window_processor = processors[args.processor_name]
args.gen_method = gen_methods[args.gen_method_name]
args.output_name = ("%s_%s_%d_%d_%s_%s_%s.h5" % (args.model_name, args.dataset, args.offset, args.length,
                                              args.apply_method_name, args.gen_method_name, args.description))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    print('loading model from {0}'.format(args.model_path))
    model = torch.load(args.model_path)
    print('loading dataset from {0}'.format(args.data_dir))
    dataset = dataset_factory[args.dataset](args.data_dir, train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    x, y, map = gen_on_set(model, loader, args.size, criterion, args.window_processor, args.gen_method,
                               args.offset, args.length, args.update_err, args.use_cuda)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    file = h5.File(os.path.join(args.output_dir, args.output_name))
    file.create_dataset("x", data=x.numpy())
    file.create_dataset("y", data=y.numpy())
    file.create_dataset("map", data=map.numpy())
    file.close()


if __name__ == "__main__":
    main()
