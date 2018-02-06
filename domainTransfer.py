

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:14 2017

@author: sebbaghs
"""

from __future__ import print_function


import sys
import os
import time
use_different_pytorch = True

import sys

if use_different_pytorch:
    sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')

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
from torchvision.utils import save_image
import argparse
#from local_models import inception_CAE_SVHN as modelFactory
import local_models



torch.manual_seed(1)

#default values
Nepochs=10000
NbatchTrain=128
NbatchTest=128
Nplot=1
Nsave=20
Nexperience=1 
learningRate=0.001
idxModel='MyDomainTransfer'
choiceLoss='L1Loss'
datasetA='mnist'
datasetB='svhn'
depth=2
lastActivation='sigmoid'



parser=argparse.ArgumentParser()
parser.add_argument('--Nepochs', default=Nepochs,type=int)
parser.add_argument('--NbatchTrain', default=NbatchTrain,type=int)
parser.add_argument('--NbatchTest', default=NbatchTest,type=int)
parser.add_argument('--Nplot', default=Nplot,type=int)
parser.add_argument('--Nsave', default=Nsave,type=int)
parser.add_argument('--Nexperience', default=Nexperience,type=int)
parser.add_argument('--learningRate', default=learningRate,type=float)
parser.add_argument('--idxModel', default=idxModel,type=str)
parser.add_argument('--choiceLoss', default=choiceLoss,type=str)
parser.add_argument('--datasetA', default=datasetA,type=str) 
parser.add_argument('--datasetB', default=datasetB,type=str) 
parser.add_argument('--depth', default=depth,type=int) 
parser.add_argument('--lastActivation', default=lastActivation,type=str) 




args = parser.parse_args()

descriptor=''
for i in vars(args):
    line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
    descriptor+=line_new
    print(line_new)
time.sleep(5)

#custom values
Nepochs=getattr(args,'Nepochs')
NbatchTrain=getattr(args,'NbatchTrain')
NbatchTest=getattr(args,'NbatchTest')
Nplot=getattr(args,'Nplot')
Nsave=getattr(args,'Nsave')
Nexperience=getattr(args,'Nexperience')
learningRate=getattr(args,'learningRate')
idxModel=getattr(args,'idxModel')
choiceLoss=getattr(args,'choiceLoss')
datasetA=getattr(args,'datasetA')
datasetB=getattr(args,'datasetB')

depth=getattr(args,'depth')
lastActivation=getattr(args,'lastActivation')

if os.name=='nt':
    modelName='{}\\local_models\\{}.py'.format(os.getcwd(),idxModel)

else:
    modelName='{}/local_models/{}.py'.format(os.getcwd(),idxModel)



class MyLoss(torch.nn.Module):
    
    def __init__(self):
        super(MyLoss,self).__init__()
        self.metrics=torch.nn.L1Loss().cuda()   

        
        
    def forward(self,inputsA, inputsB,both,codeA,codeA_INTER_B_fromA,reconstructionA,
                codeB,codeA_INTER_B_fromB,reconstructionB,
                auxCodeA,auxReconstructionA,
                auxCodeB,auxReconstructionB,
                auxCodeA_UNION_B,auxReconstrutionA_UNION_B):
        output=self.metrics(inputsA,reconstructionA)
        output+=self.metrics(inputsB,reconstructionB)
        output+=self.metrics(inputsA,auxReconstructionA)
        output+=self.metrics(inputsB,auxReconstructionB)
        
        output+=self.metrics(both,auxReconstrutionA_UNION_B)
        
        output+=self.metrics(codeA_INTER_B_fromA,auxCodeA)
        output+=self.metrics(codeA_INTER_B_fromB,auxCodeB)
        output+=torch.exp(-self.metrics(codeA_INTER_B_fromA,codeA))
        output+=torch.exp(-self.metrics(codeA_INTER_B_fromB,codeB))
        
        
       
        return (output)
    

def metrics(inp, target):
    return torch.mean(torch.abs(inp-target))
#metrics = mse_loss.cuda() 
  
def criterion(inputsA, inputsB, both,
              codeA,codeA_INTER_B_fromA,reconstructionA,
              codeB,codeA_INTER_B_fromB,reconstructionB,
              auxCodeA,auxReconstructionA,
              auxCodeB,auxReconstructionB,
              auxCodeA_UNION_B,auxReconstrutionA_UNION_B):
    output=metrics(inputsA,reconstructionA)
    output+=metrics(inputsB,reconstructionB)
    output+=metrics(inputsA,auxReconstructionA)
    output+=metrics(inputsB,auxReconstructionB)    
    output+=metrics(both,auxReconstrutionA_UNION_B)
    
    output+=metrics(codeA_INTER_B_fromA,codeB)
    output+=metrics(codeA_INTER_B_fromB,codeA:)

    temp=torch.cat([codeA_INTER_B_fromA,codeA_INTER_B_fromB],dim=0)
    output+=metrics(auxCodeA_UNION_B,temp)
    

    #output+=metrics(codeA_INTER_B_fromA,auxCodeA)
    #output+=metrics(codeA_INTER_B_fromB,auxCodeB)
    output+=torch.exp(-metrics(codeA_INTER_B_fromA,codeA))
    output+=torch.exp(-metrics(codeA_INTER_B_fromB,codeB))
    
    return(output)


#if choiceLoss=='L1Loss':
#    criterion=torch.nn.L1Loss().cuda()    
#    
#elif choiceLoss=='MSELoss':
#    criterion=torch.nn.MSELoss().cuda()    

def rescale(img):
    mi=img.min()
    ma=img.max()
    return(((img-mi)/(ma-mi)-0)*1)
    
def adaptToRGB(img):
    return(torch.cat([img,img,img],dim=0))
       
   
    

def getData(dataset):
    if dataset == "svhn":
        transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
         
         
        trainset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='train',
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)
    
        testset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='test',
                                           download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=NbatchTest,
                                             shuffle=True, num_workers=0,drop_last=True)
    elif dataset == "cifar":
        transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
        trainset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR', train=False,
                                           download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=NbatchTest,
                                             shuffle=True, num_workers=0,drop_last=True)
        
    elif dataset == "mnist":
        transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale),transforms.Lambda(adaptToRGB)])
        
        trainset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=False,
                                           download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=NbatchTest,
                                             shuffle=True, num_workers=0,drop_last=True)
    
    return(trainset,trainloader,testset,testloader)



if __name__=='__main__':
    
    #loading dataset
    
    

    trainSetA,trainLoaderA,testSetA,testLoaderA=getData(datasetA)
    trainSetB,trainLoaderB,testSetB,testLoaderB=getData(datasetB)

    NsamplesTrainA=len(trainSetA)
    NsamplesTrainB=len(trainSetB)
    NsamplesTestA=len(testSetA)
    NsamplesTestB=len(testSetB)



    
    
    print("data loaded")
    TotalTrain=min(NsamplesTrainA,NsamplesTrainB)
    TotalTest=min(NsamplesTestA,NsamplesTestB)
    
    numberOfBatchesTrain=TotalTrain//NbatchTrain
    numberOfBatchesTest=TotalTest//NbatchTest
    
    print('number of images in training set : ',TotalTrain)
    print("done in {} mini-batches of size {}".format(numberOfBatchesTrain,NbatchTrain))
    print('number of images in test set : ',TotalTest)
    print("done in {} mini-batches of size {}".format(numberOfBatchesTest,NbatchTest))
    
    
    useCuda=torch.cuda.is_available()
    
   
    
    if idxModel=='Inception_Modified':
        from local_models import Inception_Modified as modelFactory
    elif idxModel=='Resnet_Modified':
        from local_models import Resnet_Modified as modelFactory
    elif idxModel=='Unet_Modified':
        from local_models import Unet_Modified as modelFactory
    elif idxModel=='SingleCodeUnet':
        from local_models import SingleCodeUnet as modelFactory
    elif idxModel=='MyDeep':
        from local_models import MyDeep as modelFactory
    elif idxModel=='MyDomainTransfer':
        from local_models import MyDomainTransfer as modelFactory

    
    model=modelFactory.Model()
    
    
    if idxModel=='MyDeep':
        print('using MyDeep, good!') 
    else:
        print('using ',idxModel,' ...not good!')
    

    

        
    
    
    print('model loaded')
    
    
    
    
    
    #defining optimizer
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer=optim.Adadelta(model.parameters())
    optimizer=optim.Adam(model.parameters(), lr=learningRate)


    
    directory='{}/AutoEncoderForDomainTransfer/{}_{}_{}_Exp{}/'.format(os.getcwd(),idxModel,datasetA, datasetB,Nexperience)
    
    if not os.path.exists(directory):
        print('new directory : ',directory)
        
    else:
        while(os.path.exists(directory)):
            print('directory already exists : ',directory)
            Nexperience+=1
            directory='{}/AutoEncoderForDomainTransfer/{}_{}_{}_Exp{}/'.format(os.getcwd(),idxModel,datasetA, datasetB,Nexperience)
        print('new directory : ',directory)
            
    directoryData=directory+'data/'
    directoryModel=directory+'models/'
    
    #os.system('chmod 777 {}/'.format(os.path.dirname(__file__)))

    os.makedirs(directory) 
    os.makedirs(directoryData)
    os.makedirs(directoryModel)

    #save the model script in the data directory
    
    if os.name=='nt':
        commandBash='copy "{}" "{}model.py"'.format(modelName,directoryData)
    else:
        #os.system('chmod 777 {}'.format(directoryData))
        commandBash='cp {} {}model.py'.format(modelName,directoryData)
    check=os.system(commandBash)
    if check==1:
        print(commandBash)
        sys.exit("ERROR, model not copied")
        
    
    
    
    filename=directoryData+"data.txt"
    fileInfo=directoryData+"info.txt"
    
    index=2
    while(os.path.exists(filename)):
        print("file aldready existing, using a new path ",end=" ")
        filename=directoryData+"data-{}.txt".format(index)
        print(filename)
        index+=1
       
    index=2
    while(os.path.exists(fileInfo)):
        print("file aldready existing, using a new path ",end=" ")
        fileInfo=directoryData+"info-{}.txt".format(index)
        print(fileInfo)
        index+=1
        
    print('saving results at : ',filename)
    
    f= open(fileInfo,"a")
    f.write("experience done on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))
    f.write(descriptor)
    f.write("epoch,trainLoss,testLoss  \n")
    f.close()
    
    f= open(filename,"a")
    f.close()

    
    
    print('beginning of the training')
    
        
    for epoch in range(Nepochs):  # loop over the dataset multiple times
        running_loss = 0.0
        totalLoss=0.0
        for i in range(numberOfBatchesTrain):
            # get the inputs
            inputsA, labels = next(iter(trainLoaderA))
            inputsB, labels = next(iter(trainLoaderB))

            #print('shape ', inputs.size()) 
            # wrap them in Variable
            inputsA = Variable(inputsA.cuda())
            inputsB = Variable(inputsB.cuda())
            
            
            inputsA=F.pad(inputsA,(2,2,2,2))
            both=torch.cat([inputsA,inputsB],dim=0)

    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            (codeA,codeA_INTER_B_fromA,reconstructionA,codeB,codeA_INTER_B_fromB,reconstructionB,
            auxCodeA,auxReconstructionA,
            auxCodeB,auxReconstructionB,
            auxCodeA_UNION_B,auxReconstrutionA_UNION_B) = model(inputsA,inputsB,both)
                
            
            loss = criterion(inputsA, inputsB,both,codeA,codeA_INTER_B_fromA,reconstructionA,
                codeB,codeA_INTER_B_fromB,reconstructionB,
                auxCodeA,auxReconstructionA,
                auxCodeB,auxReconstructionB,
                auxCodeA_UNION_B,auxReconstrutionA_UNION_B)
            
            
            loss.backward()
            optimizer.step()
        # print statistics
            totalLoss+=loss.data[0]
            running_loss += loss.data[0]
            print('[epoch %d / %d, mini-batch %5d / %d] loss: %.3e' %(epoch + 1,Nepochs, i + 1,numberOfBatchesTrain, running_loss))
            running_loss = 0.0
            
            
            #plot_test(inputs,outputs)
        #processing test set
        testLoss=0.0
        for i in range(numberOfBatchesTest):
            
            inputsA, labels = next(iter(testLoaderA))
            inputsB, labels = next(iter(testLoaderB))

            #print('shape ', inputs.size()) 
            # wrap them in Variable
            inputsA = Variable(inputsA.cuda())
            inputsB = Variable(inputsB.cuda())
            
            inputsA=F.pad(inputsA,(2,2,2,2))
            both=torch.cat([inputsA,inputsB],dim=0)         
    
            (codeA,codeA_INTER_B_fromA,reconstructionA,codeB,codeA_INTER_B_fromB,reconstructionB,
            auxCodeA,auxReconstructionA,
            auxCodeB,auxReconstructionB,
            auxCodeA_UNION_B,auxReconstrutionA_UNION_B) = model(inputsA,inputsB,both)
                
                
            loss = criterion(inputsA, inputsB,both,codeA,codeA_INTER_B_fromA,reconstructionA,
                codeB,codeA_INTER_B_fromB,reconstructionB,
                auxCodeA,auxReconstructionA,
                auxCodeB,auxReconstructionB,
                auxCodeA_UNION_B,auxReconstrutionA_UNION_B)
            
            
            
                 
            
            
            
            
            testLoss+=loss.data[0]
            
            
            
        testLoss/=numberOfBatchesTest
        print("End of epoch ",epoch+1, ", error on test", testLoss)
        #save the data
        totalLoss/=numberOfBatchesTest
        print("End of epoch ",epoch+1," error on training set ",totalLoss ," error on test ", testLoss)

        f= open(filename,"a")
        f.write("{},{},{}  \n".format(epoch+1,totalLoss,testLoss))
        f.close()
        
        #print(images.size(), outputs.size())

        #save_image(images.data, 'real.png')
        #save_image(outputs.data, 'rec.png')

        #save the model
        if epoch%Nsave==0:
            torch.save(model.state_dict(),directoryModel+'/Epoch{}.pt'.format(epoch+1))
    #final save
    torch.save(model.state_dict(),directoryModel+'Epoch{}Final.pt'.format(epoch+1))
    
    
    
