import time

import torch
from torch.autograd import Variable
from skimage.segmentation import slic

channel_num = 1


def gen_sensitive_map_rect(model, pic, label, size, criterion, window_processor):
    """
        1. calculate the output of the picture by the net and its error w.r.t the label
        2. create a temp picture with 1 channel and the same size as the input
        3. divide the origin picture into rectangle areas
        4. for each area, process it with window_processor
            4.1 use the processed picture as network input, calculate the output and error
            4.2 if the new error is lower than the origin one, record the difference of the two errors in the temp picture, with
                with the same position as the current area
            4.3 recover the values in this area
        5. return the temp picture
    """
    map = torch.zeros(pic.size()[1:3])
    window_tensor = torch.zeros(pic.size()[1], size[0], size[1])
    curr_x = 0
    curr_y = 0
    xlimit = pic.size(1)
    ylimit = pic.size(2)
    std_out = model(pic)
    std_err = criterion(std_out, label)
    while curr_y < ylimit:
        while curr_x < xlimit:
            end_x = curr_x + size[0] if curr_x + size[0] < xlimit else xlimit
            end_y = curr_y + size[1] if curr_y + size[1] < ylimit else ylimit
            window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)] = pic.data[:, curr_x:end_x, curr_y:end_y]
            pic.data[:, curr_x:end_x, curr_y:end_y] = window_processor(pic.data[:, curr_x:end_x, curr_y:end_y])
            curr_out = model(pic)
            curr_err = criterion(curr_out, label)
            if curr_err.data[0] < std_err.data[0]:
                map[curr_x:end_x, curr_y:end_y] = std_err.data[0] - curr_err.data[0]
            pic.data[:, curr_x:end_x, curr_y:end_y] = window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)]
            curr_x += size[0]
        curr_y += size[1]
        curr_x = 0

    return map


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
    """
    map = torch.zeros((1, pic.size()[2], pic.size()[3]))
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
    while curr_y < ylimit:
        while curr_x < xlimit:
            end_x = curr_x + size[0] if curr_x + size[0] < xlimit else xlimit
            end_y = curr_y + size[1] if curr_y + size[1] < ylimit else ylimit
            window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)] = pic.data[0, :, curr_x:end_x, curr_y:end_y]
            pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_processor(pic.data[0, :, curr_x:end_x, curr_y:end_y])
            curr_out = model(pic)
            curr_err = criterion(curr_out, label)
            if curr_err.data[0] < std_err.data[0]:
                map[0, curr_x:end_x, curr_y:end_y] = std_err.data[0] - curr_err.data[0]
                if update_err:
                    std_err.data[0] = curr_err.data[0]
            else:
                pic.data[0, :, curr_x:end_x, curr_y:end_y] = window_tensor[:, 0:(end_x - curr_x), 0:(end_y - curr_y)]
            curr_x += size[0]
        curr_y += size[1]
        curr_x = 0

    return map


def gen_map_superpixel_zero(model, pic, label, size, criterion, window_processor, n_segments=100, update_err=False, use_cuda=True):
    superpixels = slic(pic, n_segments=n_segments)
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
    for i in range(n_segments):
        temp_tensor[:] = pic[:]
        for j in range(height):
            for k in range(width):
                if superpixels[j][k] == i:
                    temp_tensor[0, :, j, k] = 0
        temp_var = Variable(temp_tensor, requires_grad=False)
        if use_cuda:
            temp_var = temp_var.cuda()
        curr_out = model(temp_var)
        curr_err = criterion(curr_out, label)
        if curr_err.data[0] < std_err.data[0]:
            for j in range(height):
                for k in range(width):
                    if superpixels[j][k] == i:
                        map[0, j, k] = std_err - curr_err
            pic[:] = temp_tensor[:]
            if update_err:
                std_err.data[0] = curr_err.data[0]


def gen_on_set(model, dataset_loader, size, criterion, window_processor, gen_method, offset, length, update_err=False, use_cuda=True):
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
        one_map = gen_method(model, input, target, size, criterion, window_processor, update_err, use_cuda)
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


def dummy_model(pic):
    return torch.mean(pic)


def all_zero_processor(tensor):
    return tensor * 0


def all_one_processor(tensor):
    tensor[:] = 1
    return tensor


def avg_processor(tensor):
    tensor[:] = torch.mean(tensor)
    return tensor


gen_methods = {
    'rect_greed': gen_sensitive_map_rect_greed,
    'super_pixel_zero': gen_map_superpixel_zero,
}

processors = {
    'zero': all_zero_processor,
    'one': all_one_processor,
    'avg': avg_processor,
}


if __name__ == '__main__':
    times = 1000
    w1 = 0      # succeeded times of lowering the loss
    w2 = 0
    a1 = 0.0    # average loss ratio after processing
    a2 = 0.0
    for t in range(times):
        t1 = Variable(torch.rand((channel_num, 32, 32)))
        label = Variable(torch.zeros(1)) + 0.5
        model = dummy_model
        size = (6, 6)
        criterion = torch.nn.MSELoss()
        std_out = model(t1)
        std_err = criterion(std_out, label)
        t2 = gen_sensitive_map_rect_greed(model, t1, label, size, criterion, all_one_processor)
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
        t_gain = apply.apply_gain(t1, t2)
        t_loss = apply.apply_loss(t1, t3)
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
