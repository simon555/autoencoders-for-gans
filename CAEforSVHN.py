

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:14 2017

@author: sebbaghs
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
#from local_models import inception_CAE_SVHN as modelFactory
from local_models import Resnet_Modified as modelFactory

modelName='{}/local_models/Resnet_Modified.py'.format(os.getcwd())

torch.manual_seed(1)


Nepochs=10000
NbatchTrain=32
NbatchTest=100
Nplot=1
Nsave=10
Nexperience=1
learningRate=0.001

N1=64
N2=N1*N1




parser=argparse.ArgumentParser()
parser.add_argument('--Nepochs', default=Nepochs,type=int)
parser.add_argument('--NbatchTrain', default=NbatchTrain,type=int)
parser.add_argument('--NbatchTest', default=NbatchTest,type=int)
parser.add_argument('--Nplot', default=Nplot,type=int)
parser.add_argument('--Nsave', default=Nsave,type=int)
parser.add_argument('--Nexperience', default=Nexperience,type=int)
parser.add_argument('--learningRate', default=learningRate,type=float)

args = parser.parse_args()

descriptor=''
for i in vars(args):
    line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
    descriptor+=line_new
    print(line_new, end='')
    







if __name__=='__main__':
    
    #loading dataset
    
    def rescale(img):
        mi=img.min()
        ma=img.max()
        return(((img-mi)/(ma-mi)-0)*1)
    
       
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
                                
    trainset = torchvision.datasets.SVHN(root='./SVHN', split='train',
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.SVHN(root='./SVHN', split='test',
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=NbatchTest,
                                             shuffle=False, num_workers=0)
    
    
    
    print("data loadeid")
    TotalTrain=len(trainloader)*NbatchTrain
    TotalTest=len(testloader)*NbatchTest
    print('number of images in training set : ',TotalTrain)
    print("done in {} mini-batches of size {}".format(len(trainloader),NbatchTrain))
    print('number of images in test set : ',TotalTest)
    print("done in {} mini-batches of size {}".format(len(testloader),NbatchTest))
    
    
    useCuda=torch.cuda.is_available()
    
    #get the input channels
    imgForChannels=trainset[0][0]
    channels=imgForChannels.size()[0]
    
       
    

    model=modelFactory.ModelAE()  
    
    
    print('model loaded')
    
    
    
    
    
    #defining optimizer
    criterion=torch.nn.MSELoss().cuda()    
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer=optim.Adadelta(model.parameters())
    optimizer=optim.Adam(model.parameters(), lr=learningRate)


    
    directory='{}/results/Exp{}/'.format(os.getcwd(),Nexperience)
    
    if not os.path.exists(directory):
        print('new directory : ',directory)
        
    else:
        while(os.path.exists(directory)):
            print('directory already exists : ',directory)
            Nexperience+=1
            directory='{}/results/Exp{}/'.format(os.getcwd(),Nexperience)
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
        
    
    
    
    filename="./results/Exp{}/data/data.txt".format(Nexperience)
    index=2
    while(os.path.exists(filename)):
        print("file aldready existing, using a new path ",end=" ")
        filename="./results/Exp{}/data/data-{}.txt".format(Nexperience,index)
        print(filename)
        index+=1
        
    print('saving results at : ',filename)
    f= open(filename,"a")
    f.write("experience done on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))
    f.write(descriptor)
    f.write("epoch,trainLoss,testLoss  \n")
    f.close()
    
    
    print('beginning of the training')
    for epoch in range(Nepochs):  # loop over the dataset multiple times
        running_loss = 0.0
        totalLoss=0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print('shape ', inputs.size()) 
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        # print statistics
            totalLoss+=loss.data[0]
            running_loss += loss.data[0]
            print('[epoch %d / %d, mini-batch %5d / %d] loss: %.3e' %(epoch + 1,Nepochs, i + 1,len(trainloader), running_loss))
            running_loss = 0.0
            
            
            #plot_test(inputs,outputs)
        #processing test set
        testLoss=0.0
        for data in testloader:
            images, labels = data
            images=Variable(images.cuda())
            outputs = model(images)
            loss=criterion(outputs,images)        
            testLoss+=loss.data[0]
        testLoss/=len(testloader)
        print("End of epoch ",epoch+1, ", error on test", testLoss)
        #save the data
        totalLoss/=len(trainloader)
        print("End of epoch ",epoch+1," error on training set ",totalLoss ," error on test ", testLoss)
           
        f= open(filename,"a")
        f.write("{},{},{}  \n".format(epoch+1,totalLoss,testLoss))
        f.close()
        
        #save the model
        if epoch%Nsave==0:
            torch.save(model.state_dict(),'./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,epoch+1))
    #final save
    torch.save(model.state_dict(),'./results/Exp{}/models/Exp{}Epoch{}Final.pt'.format(Nexperience,Nexperience,epoch+1))

    
    
    
