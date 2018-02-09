import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as pl


'''Simple dense network encoders

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
from torch.nn.parameter import Parameter




class myLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        x,y=self.weight.size()
        self.factor=x*y

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight/self.factor, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class Model(nn.Module):
    def __init__(self,
                 useCuda=False,
                 inputChannel=784,
                 outputChannel=10):
        super(Model, self).__init__()

        useCuda = torch.cuda.is_available()
        self.useCuda = useCuda

        # visual parameters


        N1 = 64

        Nblocks = 64

        # first step
        self.inputChannel = inputChannel
        self.outputChannel=outputChannel

        self.layer1=myLinear(self.inputChannel,512)
        self.layer2=myLinear(512,256)
        self.layer3=myLinear(256,128)
        self.layer4=myLinear(128,64)
        self.layer5=myLinear(64,10)

        if (self.useCuda):
            self.cuda()

        print('use CUDA : ', self.useCuda)



    def forward(self, image):
        image=image.view(-1,self.inputChannel)
        image = F.relu(self.layer1(image))
        image = F.relu(self.layer2(image))
        image = F.relu(self.layer3(image))
        image = F.relu(self.layer4(image))
        image = self.layer5(image)

        return (image)
