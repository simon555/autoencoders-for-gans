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
    
    
idxModel='MyDomainTransfer_mnist_svhn_Exp11'
Epoch=181
datasetA='mnist'
datasetB='svhn'
mode='mix'
    
fileDirectory = "{}/".format(idxModel)

dataFolder=fileDirectory+'data/'
sys.path.insert(0,dataFolder)

filename=fileDirectory+'models/Epoch{}.pt'.format(Epoch)
if not os.path.exists(filename):
    print('downloading the model from remote...')
    if not os.path.exists(fileDirectory+'models/'):
        os.makedirs(fileDirectory+'models/')
    
    commandBash='pscp sebbaghs@elisa2.iro.umontreal.ca:/data/milatmp1/sebbaghs/autoencoders-for-gans/AutoEncoderForDomainTransfer/{}/models/Epoch{}.pt'.format(idxModel,Epoch)
    commandBash+=' {}'.format(filename)
    
    check=os.system(commandBash)
    print('done : ',check)
    
import model as mod

##SPECIFIC TO MYDEEP
#the_model = mod.ModelAE(depth=2,lastActivation='sigmoid')
the_model = mod.Model()



the_model.load_state_dict(torch.load(filename))
the_model.cpu()


    
def adaptToRGB(img):
    output=torch.cat([img,img,img],dim=0)
    return(output)

transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])

def getData(dataset):
    if dataset == "svhn":
        transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
         
        
        testset = torchvision.datasets.SVHN(root='../datasets/SVHN', split='test',
                                           download=True, transform=transform)
     
   
    elif dataset == "mnist":
        transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale),transforms.Lambda(adaptToRGB)])
        
        testset = torchvision.datasets.MNIST(root='../datasets/MNIST', train=False,
                                           download=True, transform=transform)
        
    return(testset)
    
    
testSetA=getData(datasetA)
testSetB=getData(datasetB)


if mode=='mix':
    for i in range(100):
        
        inputA, labels = testSetA[i]
        inputB, labels =testSetB[i]
        
        imageA=Variable(inputA.unsqueeze(0))
        imageB=Variable(inputB.unsqueeze(0))
    
        imageA=F.pad(imageA,(2,2,2,2))
        both=torch.cat([imageA,imageB],dim=0)  
        #imageB=F.pad(imageB,(2,2,2,2))
        
        
        codeA=the_model.encoderA(imageA)
        codeA_INTER_B_fromA=the_model.encoderA_INTER_B(imageA)
        
        
        codeB=the_model.encoderB(imageB)
        codeA_INTER_B_fromB=the_model.encoderA_INTER_B(imageB)
        
        fakeBfromA=the_model.decoderB(codeB,codeA_INTER_B_fromB)
        fakeAfromB=the_model.decoderA(codeA,codeA_INTER_B_fromA)

        
        
        
        imgInputA=imageA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        imgInputB=imageB[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        
        fakeA=fakeAfromB[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        fakeB=fakeBfromA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    
    
       
        #print(imgInput.mean(),' ',imgInput.std(), ' ',imgInput.max(), ' ',imgInput.min()   )
        print('image ', i)
        pl.figure(figsize=(7,5))
        
       
        pl.subplot(231)
        pl.imshow(imgInputA)
        pl.title('original image')
        
        pl.subplot(232)
        pl.imshow(imgInputB)
        pl.title("original image")
           
        
        pl.subplot(233)
        pl.imshow(fakeA)
        pl.title("fake image")
        
        pl.subplot(234)
        pl.imshow(imgInputB)
        pl.title('original image')
        
        pl.subplot(235)
        pl.imshow(imgInputA)
        pl.title("original image")
           
        
        pl.subplot(236)
        pl.imshow(fakeB)
        pl.title("fake image")
        
    
        
        pl.legend()
        
        #pl.savefig('../results/{}/image{}'.format(idxModel,i))
        pl.show()
    
if mode=='classic':
    for i in range(100):
        
        inputA, labels = testSetA[i]
        inputB, labels =testSetB[i]
        
        imageA=Variable(inputA.unsqueeze(0))
        imageB=Variable(inputB.unsqueeze(0))
    
        
        codeA,codeA_INTER_B_fromA,reconstructionA,codeB,codeA_INTER_B_fromB,reconstructionB,auxCodeA,auxReconstructionA,auxCodeB,auxReconstructionB = the_model(imageA,imageB)
        
        imgInputA=imageA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        imgInputB=imageB[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        
        outputA=reconstructionA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        outputB=reconstructionB[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    
    
        auxA=auxReconstructionA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        auxB=auxReconstructionB[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
        
        
        #print(imgInput.mean(),' ',imgInput.std(), ' ',imgInput.max(), ' ',imgInput.min()   )
        print('image ', i)
        pl.figure(figsize=(12,6))
        
       
        pl.subplot(231)
        pl.imshow(imgInputA)
        pl.title('original image')
        
        pl.subplot(232)
        pl.imshow(outputA)
        pl.title("reconstructed image")
           
        
        pl.subplot(233)
        pl.imshow(auxA)
        pl.title("aux reconstruction image")
        
        pl.subplot(234)
        pl.imshow(imgInputB)
        pl.title('original image')
        
        pl.subplot(235)
        pl.imshow(outputB)
        pl.title("reconstructed image")
           
        
        pl.subplot(236)
        pl.imshow(auxB)
        pl.title("aux reconstruction image")
        
    
        
        pl.legend()
        
        #pl.savefig('../results/{}/image{}'.format(idxModel,i))
        pl.show()
    

temp=imageA
for i in range(200):
    codeA=the_model.encoderB(imageA)
    codeA_INTER_B_fromA=the_model.encoderA_INTER_B(temp)

    temp=the_model.decoderB(codeA,codeA_INTER_B_fromA)

    out=imageA[0,:,:,:].cpu().data.numpy().transpose((1,2,0))
    pl.imshow(out)
    pl.show()
    
