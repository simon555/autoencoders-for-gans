

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




torch.manual_seed(1)

#default values
Nepochs=10000
NbatchTrain=16
NbatchTest=100
Nplot=1
Nsave=10
Nexperience=1 
learningRate=0.001
idxModel='Inception_Modified'
choiceLoss='L1Loss'




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



args = parser.parse_args()

descriptor=''
for i in vars(args):
    line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
    descriptor+=line_new
    print(line_new, end='')
    

#custom values
Nepochs=getattr(args,'Nepochs')
NbatchTrain=getattr(args,'NbatchTrain')
NbatchTest=getattr(args,'NbatchTest')
Nplot=getattr(args,'Nplot')
Nsave=getattr(args,'Nsave')
Nexperience=getattr(args,'Nexperience')
learningRate=getattr(args,'learningRate')
idxModel=getattr(args,'idxModel')
choiceLoss==getattr(args,'choiceLoss')


if os.name=='nt':
    modelName='{}\\local_models\\{}.py'.format(os.getcwd(),idxModel)

else:
    modelName='{}/local_models/{}.py'.format(os.getcwd(),idxModel)




if choiceLoss=='L1Loss':
    criterion=torch.nn.L1Loss().cuda()    
    
elif choiceLoss=='MSELoss':
    criterion=torch.nn.MSELoss().cuda()    






if __name__=='__main__':
    
    #loading dataset
    
    def rescale(img):
        mi=img.min()
        ma=img.max()
        return(((img-mi)/(ma-mi)-0)*1)
    
       
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
                                
    trainset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='train',
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='test',
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
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer=optim.Adadelta(model.parameters())
    optimizer=optim.Adam(model.parameters(), lr=learningRate)


    
    directory='{}/results/{}_Exp{}/'.format(os.getcwd(),idxModel,Nexperience)
    
    if not os.path.exists(directory):
        print('new directory : ',directory)
        
    else:
        while(os.path.exists(directory)):
            print('directory already exists : ',directory)
            Nexperience+=1
            directory='{}/results/{}Exp{}/'.format(os.getcwd(),idxModel,Nexperience)
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

    
    
    
