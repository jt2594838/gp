import torch


def apply_loss(pic, loss_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        ret[i, :, :] = pic[i, :, :] * (1 - loss_map)
    return ret


def apply_loss4D(pic, loss_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        for j in range(ret.size(1)):
            ret[i, j, :, :] = pic[i, j, :, :] * (1 - loss_map[i, :, :])
    return ret


def apply_gain(pic, gain_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        ret[i, :, :] = pic[i, :, :] + (1 - pic[i, :, :]) * gain_map
    return ret


def apply_gain4D(pic, gain_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        for j in range(ret.size(1)):
            ret[i, j, :, :] = pic[i, j, :, :] + (1 - pic[i, j, :, :]) * gain_map[i, :, :]
    return ret


def apply_avg(pic, avg_map):
    ret = torch.FloatTensor(pic.size())
    if avg_map.mean() > 0.5:
        ret[:] = torch.mean(ret)
    return ret