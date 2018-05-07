import h5py as h5
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision.transforms as transforms


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.h5_path = '/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_0.09455910949409008_unpreprocessed_VGG16_train_l5000.h5.applied'
args.show_offset = 0
args.show_length = 10


if __name__ == '__main__':
    h5_file = h5.File(args.h5_path)
    x = np.array(h5_file['x'])
    pic_cnt = x.shape[0]
    print('File contains %d pics' % pic_cnt)

    tran = transforms.ToPILImage()
    tran2 = transforms.Resize((32, 32))
    tran3 = transforms.ToTensor()

    x = x.transpose((0, 2, 3, 1))

    end = args.show_offset + args.show_length if args.show_offset + args.show_length <= pic_cnt else pic_cnt

    for i in range(args.show_offset, end):
        for j in range(3):
            x[i, :, :, j] = (x[i, :, :, j] - x[i, :, :, j].min() ) / (x[i, :, :, j].max() - x[i, :, :, j].min())

    plt.figure()

    im_height = x.shape[1]
    im_width = x.shape[2]
    im_channel = x.shape[3]
    pic = np.zeros(((end - args.show_offset) * im_height, im_width * 3, im_channel))

    for i in range(args.show_offset, end):
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), 0:im_width, :] = x[i, :, :, :]

    plt.axis('off')
    plt.imshow(pic)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

    h5_file.close()
