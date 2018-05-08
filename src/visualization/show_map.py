import h5py as h5
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision.transforms as transforms


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.map_path = '/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5'
args.applied_path = None
args.show_offset = 50
args.show_length = 10


if __name__ == '__main__':
    file = h5.File(args.map_path)
    applied_file = None
    if args.applied_path is not None:
        applied_file = h5.File(args.applied_path)
    x = np.array(file['x'])
    map = np.array(file['map'])
    applied = None
    if applied_file is not None:
        applied = np.array(applied_file['applied'])
    pic_cnt = x.shape[0]
    print('File contains %d pics' % pic_cnt)

    tran = transforms.ToPILImage()
    tran2 = transforms.Resize((32, 32))
    tran3 = transforms.ToTensor()

    x = x.transpose((0, 2, 3, 1))
    if applied is not None:
        applied = applied.transpose((0, 2, 3, 1))

    end = args.show_offset + args.show_length if args.show_offset + args.show_length <= pic_cnt else pic_cnt

    for i in range(args.show_offset, end):
        for j in range(x.shape[3]):
            x[i, :, :, j] = (x[i, :, :, j] - x[i, :, :, j].min() ) / (x[i, :, :, j].max() - x[i, :, :, j].min())
            if applied is not None:
                applied[i, :, :, j] = (applied[i, :, :, j] - applied[i, :, :, j].min()) / \
                                  (applied[i, :, :, j].max() - applied[i, :, :, j].min())
        if (map[i, :, :].max() - map[i, :, :].min()) != 0.0:
            map[i, :, :] = (map[i, :, :] - map[i, :, :].min()) / (map[i, :, :].max() - map[i, :, :].min())

    plt.figure()

    im_height = x.shape[1]
    im_width = x.shape[2]
    im_channel = x.shape[3]
    pic = np.zeros(((end - args.show_offset) * im_height, im_width * 3, im_channel))

    for i in range(args.show_offset, end):
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), 0:im_width, :] = x[i, :, :, :]
        pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), im_width:2 * im_width, 0] = map[i, :, :]
        if applied is not None:
            pic[((i - args.show_offset) * im_height):(((i - args.show_offset) + 1) * im_height), 2 * im_width:3 * im_width, :] = applied[i, :, :, :]

    pic = pic.squeeze()
    plt.axis('off')
    plt.imshow(pic, cmap='gray')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

    file.close()
    if applied_file is not None:
        applied_file.close()
