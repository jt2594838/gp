import torch

"""
This file provides process operators, which process the pixels  
Currently, we only have one (set all pixels to one) and zero (set all pixels to zero) and these operators are meant for
rectangle division method.
Using other operators may be a nice extension.
"""

def apply_zero(pic, zero_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        ret[i, :, :] = pic[i, :, :] * (1 - zero_map)
    return ret


def apply_zero4D(pic, zero_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        for j in range(ret.size(1)):
            ret[i, j, :, :] = pic[i, j, :, :] * (1 - zero_map[i, :, :])
    return ret


def apply_one(pic, one_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        ret[i, :, :] = pic[i, :, :] + (1 - pic[i, :, :]) * one_map
    return ret


def apply_one4D(pic, one_map):
    ret = torch.FloatTensor(pic.size())
    for i in range(ret.size(0)):
        for j in range(ret.size(1)):
            ret[i, j, :, :] = pic[i, j, :, :] + (1 - pic[i, j, :, :]) * one_map[i, :, :]
    return ret


apply_methods = {
    'apply_zero': apply_zero,
    'apply_zero4D': apply_zero4D,
    'apply_one': apply_one,
    'apply_one4D': apply_one4D,
}