from skimage.segmentation import slic
from dataset.factory import dataset_factory
import numpy as np
import os
import h5py as h5


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.train_dir = "./data/train_data/"
args.val_dir = "./data/val_data/"
args.dataset = 'CIFAR_10'
args.n_segments = 400
args.pixel_path = '/home/jt/codes/bs/nb/src/train/superpixel'


def gen_superpixels(dataset, n_segments):
    superpixels = None
    for i, (input, target) in enumerate(dataset):
        if superpixels is None:
            superpixels = np.zeros((len(dataset), input.shape[0], input.shape[1]))
        superpixels[i, :, :] = slic(input, n_segments=n_segments)
    return superpixels


if __name__ == '__main__':

    train_dataset = dataset_factory[args.dataset](args.train_dir, True)
    val_dataset = dataset_factory[args.dataset](args.val_dir, False)

    train_superpixels = gen_superpixels(train_dataset, args.n_segments)
    val_superpixels = gen_superpixels(val_dataset, args.n_segments)

    filename = '{0}_{1}_superpixels.h5'.format(args.dataset, args.n_segments)
    path = os.path.join(args.pixel_path, filename)
    if os.path.exists(path):
        os.remove(path)

    file = h5.File(path)
    file.create_dataset('train', data=train_superpixels)
    file.create_dataset('validate', data=val_superpixels)

    file.close()
