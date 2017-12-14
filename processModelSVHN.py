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
import results.Exp8.data.model as mod


Nexperience=8
Nepoch=251

def rescale(img):
    mi=img.min()
    ma=img.max()
    return((img-mi)/(ma-mi))
    
    

filename='./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,Nepoch)

the_model = mod.ModelAE()

the_model.load_state_dict(torch.load(filename))
the_model.cpu()

transform = transforms.Compose(
        [transforms.ToTensor()])
    
testset = torchvision.datasets.SVHN(root='./SVHN', split='test',
                                           download=True, transform=transform)

transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
    
transformed=torchvision.datasets.SVHN(root='./SVHN', split='test',
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
    pl.subplot(221)
    pl.imshow(imgInput)
    pl.subplot(222)
    pl.imshow(imgOutput)
    
    
    images= transformed[i][0]
    images=Variable(images.unsqueeze(0))
    outputs = the_model(images)
    
    imgInput2=images[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    imgOutput2=outputs[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    
    pl.subplot(223)
    pl.imshow(imgInput2)
    pl.subplot(224)
    pl.imshow(imgOutput2)
    pl.show()



images=Variable(testset[125][0].unsqueeze(0)) 
outputs = the_model(images)


imgInput=images[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
imgOutput=outputs[0,:,:,:].cpu().data.numpy().transpose((1,2,0))


pl.plot(imgOutput[15,:,0])
pl.plot(imgInput[15,:,0])
pl.show()

pl.plot(imgOutput[15,:,1])
pl.plot(imgInput[15,:,1])
pl.show()

pl.plot(imgOutput[15,:,1])
pl.plot(imgInput[15,:,1])
pl.show()
   