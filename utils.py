import torch
from torch.autograd import Variable
import os, errno


def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)



