

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


torch.manual_seed(1)


Nepochs=10000
NbatchTrain=32
NbatchTest=100
Nplot=1
Nsave=10
Nexperience=100
learningRate=0.0001

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
    






class Block(nn.Module):
    def __init__(self, Nchannels):        
        super(Block, self).__init__()
        
        self.Conv_1=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1) 
        self.BN_1=torch.nn.BatchNorm2d(Nchannels)
        self.Conv_2=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1)
        self.BN_2=torch.nn.BatchNorm2d(Nchannels)
        self.Conv_3=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1) 
        self.BN_3=torch.nn.BatchNorm2d(Nchannels)
        
       
            
    
    def forward(self,input):
        x = self.Conv_1(input)
        x = self.BN_1(x)
        x = F.relu(x)
        
        x = self.Conv_2(x)
        x = self.BN_2(x)
        x = F.relu(x)
        
        x+=input
        x = F.relu(x)

        
        x = self.Conv_3(x)
        x = self.BN_3(x)
        x = F.relu(x)
        
        return(x)

      



class Autoencoder(nn.Module):
    def __init__(self,
                 useCuda=False,
                 inputChannel=1):
        
        super(Autoencoder, self).__init__()

        self.useCuda=useCuda
        
        
        #visual parameters            
        self.pool = nn.MaxPool2d(2, 2)
        self.upSample= nn.Upsample(scale_factor=2, mode='bilinear')
        
        N1=64
        
        Nblocks=64
        
        #first step
        self.inputChannel=inputChannel
        self.Conv_1P=nn.Conv2d(self.inputChannel,N1,3,stride=1,padding=1)
        self.BN_1P=torch.nn.BatchNorm2d(N1)
        self.Conv_2P=nn.Conv2d(N1,Nblocks,3,stride=1,padding=1) 
        self.BN_2P=torch.nn.BatchNorm2d(Nblocks)
        self.Conv_3P=nn.Conv2d(Nblocks,Nblocks,3,stride=1,padding=1) 
        self.BN_3P=torch.nn.BatchNorm2d(Nblocks)
        self.Conv_4P=nn.Conv2d(Nblocks,Nblocks,3,stride=1,padding=1) 
        self.BN_4P=torch.nn.BatchNorm2d(Nblocks)
        
          
        
        


        #last step
        self.Conv_1F=nn.Conv2d(Nblocks,N1,3,stride=1,padding=1) 
        self.Conv_2F=nn.Conv2d(N1,self.inputChannel,3,stride=1,padding=1) 
        self.Conv_3F=nn.Conv2d(self.inputChannel,self.inputChannel,3,stride=1,padding=1) 
        self.Conv_4F=nn.Conv2d(self.inputChannel,self.inputChannel,3,stride=1,padding=1)       
        
        
        #encoding blocks
        self.Code_B1=Block(Nblocks)  
        self.Code_B2=Block(Nblocks)          
        self.Code_B3=Block(Nblocks)  
        self.Code_B4=Block(Nblocks)  
        self.Code_B5=Block(2*Nblocks)  
        self.Code_B6=Block(2*Nblocks)  
        self.Code_Conv1=nn.Conv2d(Nblocks,2*Nblocks,3,stride=1,padding=1)

        
        #decoding blocks
        self.DeCode_B1=Block(2*Nblocks)  
        self.DeCode_B2=Block(2*Nblocks)  
        #self.Conv_Decode1=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        self.DeCode_B3=Block(Nblocks)  
        self.DeCode_B4=Block(Nblocks)  
        #self.Conv_Decode2=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        self.DeCode_B5=Block(Nblocks)  
        self.DeCode_B6=Block(Nblocks)  
        self.DeCode_Conv1=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1)

                   
        if (self.useCuda):
            self.cuda()              
            
            #encoding blocks
            self.Code_B1.cuda()
            self.Code_B2.cuda()      
            self.Code_B3.cuda()
            self.Code_B4.cuda()
            self.Code_B5.cuda()
            self.Code_B6.cuda()
            
            #decoding blocks
            self.DeCode_B1.cuda()
            self.DeCode_B2.cuda()
            #self.Conv_Decode1.cuda()
            self.DeCode_B3.cuda()
            self.DeCode_B4
            #self.Conv_Decode2.cuda()
            self.DeCode_B5.cuda()
            self.DeCode_B6.cuda()
            
            
            
        print('use CUDA : ',self.useCuda)
        
        print('model loaded')
        

 
    def code(self,image):        
        x = self.Conv_1P(image)
        x = self.BN_1P(x)
        x = F.relu(x)
        
        x = self.Conv_2P(x)
        x = self.BN_2P(x)
        x = F.relu(x)
        
        x = self.Conv_3P(x)
        x = self.BN_3P(x)
        x = F.relu(x)
        
        x = self.Conv_4P(x)
        x = self.BN_4P(x)
        x = F.relu(x)
        
        x=self.Code_B1.forward(x)
        #x=self.Code_B2.forward(x)       
        x=self.pool(x)           


        x=self.Code_B3.forward(x)
        x=self.Code_B4.forward(x)
        

        x=self.Code_Conv1(x)        
        x=self.Code_B5.forward(x)
        x=self.Code_B6.forward(x)

        
        return(x)
        
    def decode(self,image):
         
               
        x=self.DeCode_B1.forward(image)
        x=self.DeCode_B2.forward(x)     
        x=self.DeCode_Conv1(x)        
        #x=torch.cat([x,x2],dim=1)
        #x=self.Conv_Decode1(x)
        
        x=self.DeCode_B3.forward(x)
        x=self.DeCode_B4.forward(x)   
        x=self.upSample(x)        
        #x=torch.cat([x,x1],dim=1)        
        #x=self.Conv_Decode2(x)
        
        
        x=self.DeCode_B5.forward(x)
        x=self.DeCode_B6.forward(x)
        
        x=self.Conv_1F(x)
        x = F.relu(x)
        x=self.Conv_2F(x)
        x = F.relu(x)
        x=self.Conv_3F(x)
        x = F.relu(x)
        x=self.Conv_4F(x)
        x = F.relu(x)   

        
        return(x)
        



    def forward(self,image):
        
        c=self.code(image)
        d=self.decode(c)            
        
        return(d)
        
        
    def plot_test(self,inputs,outputs):
        pl.figure()
        pl.subplot(131)
        pl.imshow(inputs.cpu().data.numpy()[0,0,:,:],cmap='gray')
        pl.subplot(132)
        pl.imshow(outputs.cpu().data.numpy()[0,0,:,:],cmap='gray')
        pl.show()
    



if __name__=='__main__':
    
    #loading dataset
    
    def rescale(img):
        mi=img.min()
        ma=img.max()
        return((img-mi)/(ma-mi))
    
       
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

    model=Autoencoder(useCuda=useCuda,
                      inputChannel=channels)   
    print('model loaded')
    
    
    
    
    
    #defining optimizer
    criterion=torch.nn.MSELoss().cuda()    
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer=optim.Adadelta(model.parameters())
    optimizer=optim.Adam(model.parameters(), lr=learningRate)


    
    directory='{}/results/Exp{}/'.format(os.path.dirname((__file__)),Nexperience)
    
    if not os.path.exists(directory):
        print('new directory : ',directory)
        
    else:
        while(os.path.exists(directory)):
            print('directory already exists : ',directory)
            Nexperience+=1
            directory='{}/results/Exp{}/'.format(os.path.dirname((__file__)),Nexperience)
        print('new directory : ',directory)
            
    directoryData=directory+'data/'
    directoryModel=directory+'models/'
    
    
    os.makedirs(directory) 
    os.makedirs(directoryData)
    os.makedirs(directoryModel)

    #save the model script in the data directory
    if os.name=='nt':
        commandBash='copy "{}" "{}model.py"'.format(__file__,directoryData)
    else:
        commandBash='cp {} {}'.format(__file__,directoryData)
    os.system(commandBash)
    
    
    
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

    
    
    
