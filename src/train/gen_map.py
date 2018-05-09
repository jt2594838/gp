import argparse
import os

import h5py as h5
import torch
import torch.nn as nn

from dataset.factory import dataset_factory
from process.apply import apply_methods
from process.generate import gen_methods, processors, gen_on_set

parser = argparse.ArgumentParser(description='Train a basic classifier')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-train_dir', type=str, default="/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_0.09455910949409008_unpreprocessed_VGG16_train_l5000.h5.applied")
parser.add_argument('-dataset', type=str, default='CIFAR_10')
parser.add_argument('-model_path', type=str, default='/home/jt/codes/bs/nb/src/train/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl')
parser.add_argument('-model_name', type=str, default="ResNet")
parser.add_argument('-use_cuda', type=bool, default=True)
parser.add_argument('-gpu_no', type=str, default='0')
parser.add_argument('-description', type=str, default='rect_quality')
parser.add_argument('-apply_method', type=str, default='apply_loss4D')
parser.add_argument('-size', type=int, default=8)
parser.add_argument('-processor_name', type=str, default='zero')
parser.add_argument('-gen_method_name', type=str, default='rect_greed')
parser.add_argument('-output_dir', type=str, default="/home/jt/codes/bs/gp/res_anzhen/train_map")
parser.add_argument('-offset', type=int, default=0)
parser.add_argument('-length', type=int, default=4300)
parser.add_argument('-update_err', type=bool, default=True)

args = parser.parse_args()
args.apply_method = apply_methods[args.apply_method]
args.size = (args.size, args.size)
args.window_processor = processors[args.processor_name]
args.gen_method = gen_methods[args.gen_method_name]
args.output_name = ("%s_%s_%d_%d_%s_%s_%s.h5" % (args.model_name, args.dataset, args.offset, args.length,
                                              args.processor_name, args.gen_method_name, args.description))


def main():
    model = torch.load(args.model_path)
    dataset = dataset_factory[args.dataset](args.train_dir, True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
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
