import h5py as h5
import numpy as np
from skimage import transform


class Arg(object):

    def __init__(self) -> None:
        super().__init__()


args = Arg()
args.data_path = '/home/jt/codes/bs/nb/src/train/data/anzhen/320x320.h5'
args.label_path = '/home/jt/codes/bs/nb/src/train/data/anzhen/targets'
args.pic_width = 224
args.pic_height = 224
args.output_path = '/home/jt/codes/bs/nb/src/train/data/anzhen/merged'

if __name__ == '__main__':
    data_file = h5.File(args.data_path)
    label_file = h5.File(args.label_path)

    label_image_id = label_file['image_id'][:]
    label_y = label_file['y'][:]

    data_image_id = data_file['image_id'][:]
    data_x = data_file['X']

    output_x = np.zeros((label_image_id.shape[0], args.pic_height, args.pic_width))
    output_y = np.zeros((label_image_id.shape[0], 1))

    index = 0
    for i in range(label_image_id.shape[0]):
        for j in range(data_image_id.shape[0]):
            if label_image_id[i] == data_image_id[j]:
                output_x[index, :, :] = transform.resize(data_x[j, :, :, 0], (args.pic_height, args.pic_width))
                output_y[index, :] = np.argmax(label_y[i, :], 0)
                index += 1
                break

    output_file = h5.File(args.output_path)
    output_file.create_dataset('x', data=output_x)
    output_file.create_dataset('y', data=output_y)
    output_file.close()
