# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:30:03 2017

@author: simon
"""
import sys
import os
currentDirectory = os.getcwd()
if not currentDirectory in sys.path:
    print('adding local directory : ', currentDirectory)
    sys.path.insert(0,currentDirectory)


import torch
import test
import matplotlib.pyplot as pl
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import time
import torchvision
import torchvision.transforms as transforms
import os


Nexperience=10
Nepoch=21


filename='./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,Nepoch)

the_model = test.Autoencoder()

the_model.load_state_dict(torch.load(filename))
the_model.cpu()

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    
    
testset = torchvision.datasets.MNIST(root='./MNIST', train=False,
                                           download=True, transform=transform)

    
for data in testset:
    images, labels = data
    images=Variable(images.unsqueeze(0))
    outputs = the_model.code(images)
    the_model.plot_test(images,outputs)