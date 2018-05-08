import os

import h5py as h5
import torch
import torch.nn as nn

import dataset.factory as df
import process.generate as gen

dataset_factory = df.dataset_factory


class Arg(object):

    def __init__(self):
        super().__init__()


args = Arg()
args.model_path = "/home/jt/codes/bs/gp/res_anzhen/original_model/ResNet101_anzhen_3_200_98.35051569987819.pkl"
args.model_name = "ResNet"
args.dataset = "anzhen"
args.train_dir = "/home/jt/codes/bs/gp/data/anzhen/merged2"
args.batch_size = 1
args.workers = 1
args.size = (8, 8)
args.window_processor = gen.all_zero_processor
args.processor_name = 'zero'
args.gen_method = gen.gen_sensitive_map_rect_greed
args.gen_method_name = 'greed'
args.output_dir = "/home/jt/codes/bs/gp/res_anzhen/train_map"
args.offset = 0
args.length = 4300
args.description = 'rect_quality'
args.output_name = ("%s_%s_%d_%d_%s_%s_%s.h5" % (args.model_name, args.dataset, args.offset, args.length,
                                              args.processor_name, args.gen_method_name, args.description))
args.use_cuda = True
args.update_err = True
args.gpu_no = "0"


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

    x, y, map = gen.gen_on_set(model, loader, args.size, criterion, args.window_processor, args.gen_method,
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
