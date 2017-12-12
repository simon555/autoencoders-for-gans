# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:15:51 2017

@author: simon
"""

import sys
import os

currentDirectory = os.getcwd()
if not currentDirectory in sys.path:
    print('adding local directory : ', currentDirectory)
    sys.path.insert(0,currentDirectory)

import torch
import matplotlib.pyplot as pl
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import argparse
import inceptionEncoder
import inceptionDecoder


class inceptionAE(nn.Module):
    def __init__(self,
                 useCuda=False,
                 pretrained=True):
        super(inceptionAE, self).__init__()
        self.encoder=inceptionEncoder.encoder()
        self.decoder=inceptionDecoder.decoder()        
        self.useCuda=torch.cuda.is_available()
        
        if self.useCuda:
            self.encoder.cuda()
            self.decoder.cuda()
        
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded')
        
    def forward(self,image):
        return(self.decoder(self.encoder(image)))
        

            
        
        
        
        
        
        
        
        