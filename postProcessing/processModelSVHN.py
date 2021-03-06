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


def rescale(img):
    mi=img.min()
    ma=img.max()
    return((img-mi)/(ma-mi))
    
modelName='Resnet_Simple'
idxModel=modelName+'_cifar_Exp1'

dataset='cifar'
    
fileDirectory = os.path.join(currentDirectory,"..", "results","{}".format(idxModel))

dataFolder=os.path.join(fileDirectory,'data')

filename=os.path.join(fileDirectory,'models','bestTestLoss.pt')
if not os.path.exists(filename):
    print('downloading the model from remote...')
    if not os.path.exists(os.path.join(fileDirectory,'models')):
        os.makedirs(os.path.join(fileDirectory,'models'))
    
    commandBash='pscp sebbaghs@elisa2.iro.umontreal.ca:/data/milatmp1/sebbaghs/autoencoders-for-gans/results/{}/models/bestTestLoss.pt'.format(idxModel)
    commandBash+=' {}'.format(filename)
    
    check=os.system(commandBash)
    print('done : ',check)
    
sys.path.insert(0,'C:/Users/simon/Desktop/MILA/autoencoders-for-gans/postProcessing/../results/{}/data'.format(idxModel))

import model as mod

print('model downloaded')
##SPECIFIC TO MYDEEP
#the_model = mod.ModelAE(depth=2,lastActivation='sigmoid')
the_model = mod.ModelAE()



the_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
the_model.cpu()


transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])


if dataset == "cifar":
        testset = torchvision.datasets.CIFAR10(root='../datasets/CIFAR', train=False,
                                           download=True, transform=transform)
        
elif dataset=='svhn':
    testset = torchvision.datasets.SVHN(root='../datasets/SVHN', split='test',
                                           download=True, transform=transform)

    
for i,data in enumerate(testset):
    images, labels = data
    images=Variable(images.unsqueeze(0))
    outputs = the_model(images)
    
    imgInput=images[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    imgOutput=outputs[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    
    #print(imgInput.mean(),' ',imgInput.std(), ' ',imgInput.max(), ' ',imgInput.min()   )
    print('image ', i)
    pl.figure()
    
    if i==24:
        inp=imgInput
        out=imgOutput
    
    pl.subplot(131)
    pl.imshow(imgInput)
    pl.title('original image')
    pl.subplot(132)
    pl.imshow(imgOutput)
    pl.title("reconstructed image")

    error=np.square(np.subtract(imgInput, imgOutput)).mean(axis=2)
    pl.subplot(133)
    pl.imshow(error)
    pl.title("error field")
    
    pl.colorbar()
    pl.legend()
    
    #pl.savefig('../results/{}/image{}'.format(idxModel,i))
    pl.show()


