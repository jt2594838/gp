import time

import numpy as np
import torch
from skimage.segmentation import slic
from torch.autograd import Variable

"""
Belows are different methods to generate a process map from an input given a trained network.
The main difference here is the segmentation method. We have adopted 2 typical methods, simple rectangular segmentation
and SLIC superpixel segmentation. Other changes may be randomly choose the segments when generating process maps.
You may try other segmentation methods or find better ways to utilize current methods.
"""


def gen_sensitive_map_rect_greed(model, pic, label, size, criterion, window_processor, update_err=False, use_cuda=True):
    """
        1. calculate the output of the picture by the net and its error w.r.t the label
        2. create a temp picture with 1 channel and the same size as the input
        3. divide the origin picture into rectangle areas
        4. for each area, process it with window_processor
            4.1 use the processed picture as network input, calculate the output and error
            4.2 if the new error is lower than the origin one, record the difference of the two errors in the temp picture, with
                with the same position as the current area and update the standard error if needed
            4.3 else recover the values in this area
        5. return the temp picture
    :param model:
        The classifier model which evaluates the effect of process.
    :param pic:
        From which the process map (non-critical area) is generated. A 4d array of size [1, depth, height, width].
    :param label:
        The true label of pic.
    :param size:
        The size of the rectangle used to detect non-critical area. A tuple of two elements (height, width).
    :param criterion:
        The method of calculating classification errors, e.g., MSE, cross entropy.
    :param window_processor:
        How to process each segment, e.g., set to one or set to zero.
    :param update_err:
        If set to true, then after each successful process (classification error is less than  the standard error), the
        standard error will be set to the new less error, which means we can find more effective and less non-critical
        area.
    :param use_cuda:
        Whether to use GPU or not.
    :return:
        The process map.
    """
    map = torch.zeros((1, pic.size()[2], pic.size()[3]))
    # a temporary variable to store the pixels in a segmentation.
    window_tensor = torch.zeros(pic.size()[1], size[0], size[1])
    curr_x = 0
    curr_y = 0
    xlimit = pic.size(2)
    ylimit = pic.size(3)
    label = Variable(label, requires_grad=False)
    pic = Variable(pic, requires_grad=False)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    std_out = model(pic)
    std_err = criterion(std_out, label)
    # for each segment
    while curr_y < ylimit:
        while curr_x < xlimit:
            end_x = curr_x + size[0] if curr_x + size[0] < xlimit else xlimit
            end_y = curr_y + size[1] if curr_y + size[1] < ylimit else ylimit
            # save the segment
            window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)] = pic.data[0, :, curr_x:end_x, curr_y:end_y]
            # process the segment
            pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_processor(pic.data[0, :, curr_x:end_x, curr_y:end_y])
            # re-calculate the error
            curr_out = model(pic)
            curr_err = criterion(curr_out, label)
            # if the error decreases, accept this process and record positions being processed
            if curr_err.data[0] < std_err.data[0]:
                map[0, curr_x:end_x, curr_y:end_y] = std_err.data[0] - curr_err.data[0]
                # if necessary, update the error to minimize the error
                if update_err:
                    std_err.data[0] = curr_err.data[0]
            # restore the segment otherwise
            else:
                pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)]
            curr_x += size[0]
        curr_y += size[1]
        curr_x = 0

    return map


def gen_sensitive_map_rect_greed_rnd(model, pic, label, size, criterion, window_processor, update_err=False, use_cuda=True):
    """
        1. calculate the output of the picture by the net and its error w.r.t the label
        2. create a temp picture with 1 channel and the same size as the input
        3. divide the origin picture into rectangle areas
        4. choose a random area, process it with window_processor
            4.1 use the processed picture as network input, calculate the output and error
            4.2 if the new error is lower than the origin one, record the difference of the two errors in the temp picture, with
                with the same position as the current area and update the standard error if needed
            4.3 else recover the values in this area
        5. return the temp picture
    :param model:
        The classifier model which evaluates the effect of process.
    :param pic:
        From which the process map (non-critical area) is generated. A 4d array of size [1, depth, height, width].
    :param label:
        The true label of pic.
    :param size:
        The size of the rectangle used to detect non-critical area. A tuple of two elements (height, width).
    :param criterion:
        The method of calculating classification errors, e.g., MSE, cross entropy.
    :param window_processor:
        How to process each segment, e.g., set to one or set to zero.
    :param update_err:
        If set to true, then after each successful process (classification error is less than  the standard error), the
        standard error will be set to the new less error, which means we can find more effective and less non-critical
        area.
    :param use_cuda:
        Whether to use GPU or not.
    :return:
        The process map.
    """
    map = torch.zeros((1, pic.size()[2], pic.size()[3]))
    # a temporary variable to store the pixels in a segmentation.
    window_tensor = torch.zeros(pic.size()[1], size[0], size[1])
    xlimit = pic.size(2)
    ylimit = pic.size(3)
    label = Variable(label, requires_grad=False)
    pic = Variable(pic, requires_grad=False)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    std_out = model(pic)
    std_err = criterion(std_out, label)
    x_num = int(np.ceil(xlimit / size[0]))
    y_num = int(np.ceil(ylimit / size[1]))
    # for each segment
    for i in np.random.choice(x_num, x_num, replace=False):
        for j in np.random.choice(y_num, y_num, replace=False):
            curr_x = i * size[0]
            curr_y = j * size[1]
            end_x = curr_x + size[0] if curr_x + size[0] < xlimit else xlimit
            end_y = curr_y + size[1] if curr_y + size[1] < ylimit else ylimit
            # save the segment
            window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)] = pic.data[0, :, curr_x:end_x, curr_y:end_y]
            # process the segment
            pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_processor(pic.data[0, :, curr_x:end_x, curr_y:end_y])
            # re-calculate the error
            curr_out = model(pic)
            curr_err = criterion(curr_out, label)
            # if the error decreases, accept this process and record positions being processed
            if curr_err.data[0] < std_err.data[0]:
                map[0, curr_x:end_x, curr_y:end_y] = std_err.data[0] - curr_err.data[0]
                # if necessary, update the error to minimize the error
                if update_err:
                    std_err.data[0] = curr_err.data[0]
            # restore the segment otherwise
            else:
                pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)]
            curr_x += size[0]
    return map


def gen_map_superpixel_zero(model, pic, label, size, criterion, update_err=False, use_cuda=True,
                            compactness=0.03, **kwargs):
    """
    Like gen_sensitive_map_rect_greed, but the segmentation method is SLIC and the processor is a zero_processor.
     :param model:
        The classifier model which evaluates the effect of process.
    :param pic:
        From which the process map (non-critical area) is generated. A 4d array of size [1, depth, height, width].
    :param label:
        The true label of pic.
    :param size:
        The number of superpixels.
    :param criterion:
        The method of calculating classification errors, e.g., MSE, cross entropy.
    :param window_processor:
        How to process each segment, e.g., set to one or set to zero.
    :param update_err:
        If set to true, then after each successful process (classification error is less than  the standard error), the
        standard error will be set to the new less error, which means we can find more effective and less non-critical
        area.
    :param use_cuda:
        Whether to use GPU or not.
    :param compactness:
        Segmentation method parameter.
    :param kwargs:
    :return:
        The process map.
    """
    n_segments = size[0]
    superpixels = slic(pic.squeeze(), n_segments=n_segments, compactness=compactness)
    height = pic.size()[2]
    width = pic.size()[3]
    map = torch.zeros((1, height, width))
    temp_tensor = torch.zeros(pic.size())
    label = Variable(label, requires_grad=False)
    pic = Variable(pic, requires_grad=False)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    std_out = model(pic)
    std_err = criterion(std_out, label)
    # for each segment
    for i in range(n_segments):
        # copy the input
        temp_tensor[:] = pic.data[:]
        # process the segment
        for j in range(height):
            for k in range(width):
                if superpixels[j][k] == i:
                    temp_tensor[0, :, j, k] = 0
        temp_var = Variable(temp_tensor, requires_grad=False)
        if use_cuda:
            temp_var = temp_var.cuda()
        # re-calculate the error
        curr_out = model(temp_var)
        curr_err = criterion(curr_out, label)
        # if the error decreases, accept this process and record positions being processed
        if curr_err.data[0] < std_err.data[0]:
            for j in range(height):
                for k in range(width):
                    if superpixels[j][k] == i:
                        map[0, j, k] = std_err.data[0] - curr_err.data[0]
            pic.data[:] = temp_tensor[:]
            # if necessary, update the error to minimize the error
            if update_err:
                std_err.data[0] = curr_err.data[0]
    return map


def gen_map_superpixel_zero_rnd(model, pic, label, size, criterion, update_err=False, use_cuda=True,
                                compactness=0.03, **kwargs):
    """
    Like gen_sensitive_map_rect_greed, but the segmentation method is SLIC and the processor is a zero_processor and
    the segments are randomly chosen.
     :param model:
        The classifier model which evaluates the effect of process.
    :param pic:
        From which the process map (non-critical area) is generated. A 4d array of size [1, depth, height, width].
    :param label:
        The true label of pic.
    :param size:
        The number of superpixels.
    :param criterion:
        The method of calculating classification errors, e.g., MSE, cross entropy.
    :param window_processor:
        How to process each segment, e.g., set to one or set to zero.
    :param update_err:
        If set to true, then after each successful process (classification error is less than  the standard error), the
        standard error will be set to the new less error, which means we can find more effective and less non-critical
        area.
    :param use_cuda:
        Whether to use GPU or not.
    :param compactness:
        Segmentation method parameter.
    :param kwargs:
    :return:
        The process map.
    """
    n_segments = size[0]
    superpixels = slic(pic.squeeze(), n_segments=n_segments, compactness=compactness)
    height = pic.size()[2]
    width = pic.size()[3]
    map = torch.zeros((1, height, width))
    temp_tensor = torch.zeros(pic.size())
    label = Variable(label, requires_grad=False)
    pic = Variable(pic, requires_grad=False)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    std_out = model(pic)
    std_err = criterion(std_out, label)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()

    seg_list = {}
    for i in n_segments:
        seg_list[str(i)] = []
    for i in range(height):
        for j in range(width):
            seg_list[str(superpixels[i, j])].append((i, j))
    # for each segment
    for i in np.random.choice(n_segments, n_segments, replace=False):
        seg_coords = seg_list[str(i)]
        # copy the segment and process the segment
        for (j, k) in seg_coords:
            temp_tensor[0, :, j, k] = pic.data[0, :, j, k]
            pic.data[0, :, j, k] = 0
        temp_var = Variable(pic, requires_grad=False)
        if use_cuda:
            temp_var = temp_var.cuda()
        # re-calculate the error
        curr_out = model(temp_var)
        curr_err = criterion(curr_out, label)
        # if the error decreases, accept this process and record positions being processed
        if curr_err.data[0] < std_err.data[0]:
            for (j, k) in seg_coords:
                map[0, j, k] = std_err.data[0] - curr_err.data[0]
            # if necessary, update the error to minimize the error
            if update_err:
                std_err.data[0] = curr_err.data[0]
        # restore the segment otherwise
        else:
            for (j, k) in seg_coords:
                pic.data[0, :, j, k] = temp_tensor[0, :, j, k]

    return map


def gen_map_superpixel_one(model, pic, label, size, criterion, window_processor, update_err=False, use_cuda=True):
    """
        Like gen_sensitive_map_rect_greed, but the segmentation method is SLIC and the processor is a one_processor.
         :param model:
            The classifier model which evaluates the effect of process.
        :param pic:
            From which the process map (non-critical area) is generated. A 4d array of size [1, depth, height, width].
        :param label:
            The true label of pic.
        :param size:
            The number of superpixels.
        :param criterion:
            The method of calculating classification errors, e.g., MSE, cross entropy.
        :param window_processor:
            How to process each segment, e.g., set to one or set to zero.
        :param update_err:
            If set to true, then after each successful process (classification error is less than  the standard error), the
            standard error will be set to the new less error, which means we can find more effective and less non-critical
            area.
        :param use_cuda:
            Whether to use GPU or not.
        :param compactness:
            Segmentation method parameter.
        :param kwargs:
        :return:
            The process map.
        """
    n_segments = size[0]
    superpixels = slic(pic.squeeze(), n_segments=n_segments, compactness=0.03)
    height = pic.size()[2]
    width = pic.size()[3]
    map = torch.zeros((1, height, width))
    temp_tensor = torch.zeros(pic.size())
    label = Variable(label, requires_grad=False)
    pic = Variable(pic, requires_grad=False)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    std_out = model(pic)
    std_err = criterion(std_out, label)
    if use_cuda:
        label = label.cuda()
        pic = pic.cuda()
    # for each segment
    for i in range(n_segments):
        # copy the input
        temp_tensor[:] = pic.data[:]
        # process the segment
        for j in range(height):
            for k in range(width):
                if superpixels[j][k] == i:
                    temp_tensor[0, :, j, k] = 1
        temp_var = Variable(temp_tensor, requires_grad=False)
        if use_cuda:
            temp_var = temp_var.cuda()
        # re-calculate the error
        curr_out = model(temp_var)
        curr_err = criterion(curr_out, label)
        # if the error decreases, accept this process and record positions being processed
        if curr_err.data[0] < std_err.data[0]:
            for j in range(height):
                for k in range(width):
                    if superpixels[j][k] == i:
                        map[0, j, k] = std_err.data[0] - curr_err.data[0]
            pic.data[:] = temp_tensor[:]
            # if necessary, update the error to minimize the error
            if update_err:
                std_err.data[0] = curr_err.data[0]
    return map


def gen_on_set(model, dataset_loader, size, criterion, window_processor, gen_method, offset, length, update_err=False, use_cuda=True):
    """
    Generate a process map for every input in given dataset.
    :param model:
        The classifier model which evaluates the effect of process.
    :param dataset_loader:
        From which the process map (non-critical area) is generated. Each input should be a 4d array
        of size [1, depth, height, width].
    :param size:
        A tuple of two elements (height, width) when using rectangular segmentation or the number of superpixels if
        superpixel segmentation is used.
    :param criterion:
        The method of calculating classification errors, e.g., MSE, cross entropy.
    :param window_processor:
        How to process each segment, e.g., set to one or set to zero.
    :param gen_method:
        The segmentation method.
    :param offset:
        The offset of the dataset.
    :param length:
        How many inputs should be used to generate process map.
    :param update_err:
            If set to true, then after each successful process (classification error is less than  the standard error), the
            standard error will be set to the new less error, which means we can find more effective and less non-critical
            area.
    :param use_cuda:
            Whether to use GPU or not.
    :return:
        inputs, labels, process maps
    """
    print('Map generation on dataset begins..., dataset size %d, offset %d, length %d' %
          (len(dataset_loader), offset, length))
    end = time.time()
    x = None
    y = None
    map = None
    for i, (input, target) in enumerate(dataset_loader):
        if i < offset:
            continue
        if i - offset >= length:
            break
        one_map = gen_method(model, input, target, size,
                             criterion=criterion, window_processor=window_processor, update_err=update_err, use_cuda=use_cuda)
        # normalization
        max_ele = torch.max(one_map)
        if max_ele != 0:
            one_map = one_map / max_ele
        # update new dataset
        if map is None:
            map = torch.zeros((length, one_map.size(1), one_map.size(2)))
            map[i, :, :] = one_map[0, :, :]
        else:
            map[i, :, :] = one_map[0, :, :]
        if x is None:
            x = torch.zeros((length, input.size(1), input.size(2), input.size(3)))
            x[i, :, :, :] = input[0, :, :, :]
        else:
            x[i, :, :, :] = input[0, :, :, :]
        if y is None:
            y = torch.zeros((length, ))
            y[i] = target[0]
        else:
            y[i] = target[0]
        if (i + 1) % 10 == 0:
            print("%d maps generated" % (i+1))

    print('Map generation on dataset ends after %f s' % (time.time() - end))
    return x, y, map


gen_methods = {
    'rect_greed': gen_sensitive_map_rect_greed,
    'rect_rnd': gen_sensitive_map_rect_greed_rnd,
    'super_pixel_zero': gen_map_superpixel_zero,
    'super_pixel_zero_rnd': gen_sensitive_map_rect_greed_rnd,
    'super_pixel_one': gen_map_superpixel_one,
}


def all_zero_processor(tensor):
    return tensor * 0


def all_one_processor(tensor):
    tensor[:] = 1
    return tensor


processors = {
    'zero': all_zero_processor,
    'one': all_one_processor,
}


def dummy_model(pic):
    return torch.mean(pic)


def test():
    channel_num = 1
    times = 1000
    w1 = 0  # succeeded times of lowering the loss
    w2 = 0
    a1 = 0.0  # average loss ratio after processing
    a2 = 0.0
    for t in range(times):
        t1 = torch.rand((1, channel_num, 224, 224))
        label = torch.zeros(1) + 0.5
        model = dummy_model
        size = (6, 6)
        criterion = torch.nn.MSELoss()
        std_out = model(Variable(t1))
        std_err = criterion(std_out, Variable(label))
        t2 = gen_map_superpixel_zero(model, t1, label, size, criterion, all_one_processor)
        if torch.max(t2) != 0:
            t2 = t2 / torch.max(t2)
        t3 = gen_sensitive_map_rect_greed(model, t1, label, size, criterion, all_zero_processor)
        if torch.max(t3) != 0:
            t3 = t3 / torch.max(t3)

        for i in range(32):
            for j in range(32):
                if t2[i, j] < 0.98:
                    t2[i, j] = 0

        for i in range(32):
            for j in range(32):
                if t3[i, j] < 0.98:
                    t3[i, j] = 0

        import process.apply as apply
        t_gain = apply.apply_one(t1, t2)
        t_loss = apply.apply_zero(t1, t3)
        # print(t1, t2, t_gain)
        t_gain_out = model(Variable(t_gain))
        t_loss_out = model(Variable(t_loss))
        t_gain_error = criterion(t_gain_out, label)
        t_loss_error = criterion(t_loss_out, label)
        # print(torch.nonzero(t2).size(), torch.nonzero(t3).size())
        # print(torch.mean(t1).data[0], std_err.data[0], t_gain_error.data[0] / std_err.data[0], t_loss_error.data[0] / std_err.data[0])
        if std_err.data[0] > t_gain_error.data[0]:
            w1 += 1
        a1 += t_gain_error.data[0] / std_err.data[0]
        if std_err.data[0] > t_loss_error.data[0]:
            w2 += 1
        a2 += t_loss_error.data[0] / std_err.data[0]
    print(w1, w2, a1 / times, a2 / times)


def test_slic():
    import h5py as h5
    file = h5.File('/home/jt/codes/bs/gp/data/anzhen/merged2')
    pics = file['x'][:]
    pic = pics[0, :, :]
    segments = slic(pic, n_segments=200)
    segments2 = slic(torch.from_numpy(pic), n_segments=200)
    print(segments)
    print(segments2)


if __name__ == '__main__':
    test_slic()
