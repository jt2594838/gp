import h5py as h5
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision.transforms as transforms


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.map_path = '/home/jt/codes/bs/nb/src/train/maps/ResNet_CIFAR_10_0_10_avg_greed.h5'
args.applied_path = '/home/jt/codes/bs/nb/src/train/maps/ResNet_CIFAR_10_0_10_avg_greed.h5.applied_avg'
args.show_offset = 0
args.show_length = 10


if __name__ == '__main__':
    file = h5.File(args.map_path)
    applied_file = h5.File(args.applied_path)
    x = np.array(file['x'])
    map = np.array(file['map'])
    applied = np.array(applied_file['applied'])
    pic_cnt = x.shape[0]
    print('File contains %d pics' % pic_cnt)

    tran = transforms.ToPILImage()
    tran2 = transforms.Resize((32, 32))
    tran3 = transforms.ToTensor()

    x = x.transpose((0, 2, 3, 1))
    applied = applied.transpose((0, 2, 3, 1))

    end = args.show_offset + args.show_length if args.show_offset + args.show_length <= pic_cnt else pic_cnt

    for i in range(args.show_offset, end):
        for j in range(3):
            x[i, :, :, j] = (x[i, :, :, j] - x[i, :, :, j].min() ) / (x[i, :, :, j].max() - x[i, :, :, j].min())
            applied[i, :, :, j] = (applied[i, :, :, j] - applied[i, :, :, j].min()) / \
                                  (applied[i, :, :, j].max() - applied[i, :, :, j].min())
        map[i, :, :] = (map[i, :, :] - map[i, :, :].min()) / (map[i, :, :].max() - map[i, :, :].min())

    plt.figure()

    im_height = x.shape[1]
    im_width = x.shape[2]
    im_channel = x.shape[3]
    pic = np.zeros(((end - args.show_offset) * im_height, im_width * 3, im_channel))

    for i in range(args.show_offset, end):
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), 0:im_width, :] = x[i, :, :, :]
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), im_width:2 * im_width, 0] = map[i, :, :]
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), 2 * im_width:3 * im_width, :] = applied[i, :, :, :]

    plt.axis('off')
    plt.imshow(pic)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

    file.close()
    applied_file.close()
