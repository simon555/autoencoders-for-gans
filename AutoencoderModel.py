

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:14 2017

@author: sebbaghs
"""

import torch
import matplotlib.pyplot as pl
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import timeit
import torchvision
import torchvision.transforms as transforms
import os


print("using CUDA : ",torch.cuda.is_available())

torch.cuda


print("modules loaded ")
torch.manual_seed(1)

Nepochs=100
NbatchTrain=50
NbatchTest=100

Nexperience=7



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./MNIST', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./MNIST', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=NbatchTest,
                                         shuffle=False, num_workers=2)

print("data loadeid")
TotalTrain=len(trainloader)*NbatchTrain
TotalTest=len(testloader)*NbatchTest
print('number of images in training set : ',TotalTrain)
print("done in {} mini-batches of size {}".format(len(trainloader),NbatchTrain))
print('number of images in test set : ',TotalTest)
print("done in {} mini-batches of size {}".format(len(testloader),NbatchTest))

N1=64
N2=N1*N1
learningRate=0.001


class Block(nn.Module):
    def __init__(self, Nchannels):        
        super(Block, self).__init__()
        self.Conv_1=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1) 
        self.BN_1=torch.nn.BatchNorm2d(Nchannels,affine=False)
        self.Conv_2=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1)
        self.BN_2=torch.nn.BatchNorm2d(Nchannels,affine=False)
        self.Conv_3=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1) 
        self.BN_3=torch.nn.BatchNorm2d(Nchannels,affine=False)
    
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
                 input_shape=(32,32,3)):
        
        super(Autoencoder, self).__init__()

       
        #visual parameters
        self.pool = nn.MaxPool2d(2, 2)
        self.upSample= nn.Upsample(scale_factor=2, mode='bilinear')
        self.input_shape=input_shape
        
        N1=32
        
        Nblocks=64
        
        #first step

        self.Conv_1P=nn.Conv2d(3,N1,3,stride=1,padding=1)
        self.BN_1P=torch.nn.BatchNorm2d(N1)
        self.Conv_2P=nn.Conv2d(N1,Nblocks,3,stride=1,padding=1) 
        self.BN_2P=torch.nn.BatchNorm2d(Nblocks)
        self.Conv_3P=nn.Conv2d(Nblocks,Nblocks,3,stride=1,padding=1) 
        self.BN_3P=torch.nn.BatchNorm2d(Nblocks)
        self.Conv_4P=nn.Conv2d(Nblocks,Nblocks,3,stride=1,padding=1) 
        self.BN_4P=torch.nn.BatchNorm2d(Nblocks)
        
        #encoding blocks
        self.Code_B1=Block(Nblocks)
        self.Code_B2=Block(Nblocks)        
        self.Code_B3=Block(Nblocks)
        self.Code_B4=Block(Nblocks)
        self.Code_B5=Block(Nblocks)
        self.Code_B6=Block(Nblocks)


        
        #decoding blocks
        self.DeCode_B1=Block(Nblocks)
        self.DeCode_B2=Block(Nblocks)
        self.Conv_Decode1=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        self.DeCode_B3=Block(Nblocks)
        self.DeCode_B4=Block(Nblocks)
        self.Conv_Decode2=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        self.DeCode_B5=Block(Nblocks)
        self.DeCode_B6=Block(Nblocks)


        
        #last step
        self.Conv_1F=nn.Conv2d(Nblocks,N1,3,stride=1,padding=1) 
        self.Conv_2F=nn.Conv2d(N1,3,3,stride=1,padding=1) 
        self.Conv_3F=nn.Conv2d(3,3,3,stride=1,padding=1) 
        self.Conv_4F=nn.Conv2d(3,3,3,stride=1,padding=1)       
        

 



    def forward(self,image):
        
# =============================================================================
#         Pre Step
# =============================================================================
        
       
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

# =============================================================================
#       CODE
# =============================================================================
        x=self.Code_B1.forward(x)
        x1=self.Code_B2.forward(x)
        
        x=self.pool(x1)           
        x=self.Code_B3.forward(x)
        x2=self.Code_B4.forward(x)
        
        x=self.pool(x2)        
        x=self.Code_B5.forward(x)
        x=self.Code_B6.forward(x)


        
        
# =============================================================================
#         DECODE
# =============================================================================
        x=self.DeCode_B1.forward(x)
        x=self.DeCode_B2.forward(x)     
        x=self.upSample(x)        
        x=torch.cat([x,x2],dim=1)
        x=self.Conv_Decode1(x)
        
        x=self.DeCode_B3.forward(x)
        x=self.DeCode_B4.forward(x)   
        x=self.upSample(x)        
        x=torch.cat([x,x1],dim=1)        
        x=self.Conv_Decode2(x)
        
        
        x=self.DeCode_B5.forward(x)
        x=self.DeCode_B6.forward(x)

        
# =============================================================================
#         Last step
# =============================================================================
        
        x=self.Conv_1F(x)
        x = F.relu(x)
        x=self.Conv_2F(x)
        x = F.relu(x)
        x=self.Conv_3F(x)
        x = F.relu(x)
        x=self.Conv_4F(x)
        output = F.relu(x)       
        
        
        return(output)



x=Variable(torch.randn(1,3,32,32))
model=Autoencoder()
y=model(x)
model.cuda()
print('model loaded')


print(y.size())
import torch.optim as optim
criterion=torch.nn.MSELoss().cuda()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer=optim.Adam(model.parameters(), lr=learningRate)
#optimizer=optim.Adadelta(model.parameters())
filename="./results/Exp{}/data/data_lr-{}.txt".format(Nexperience,learningRate)
index=2
while(os.path.exists(filename)):
    print("file aldready existing, using a new path ",end=" ")
    filename="./results/Exp{}/data/data-{}.txt".format(Nexperience,index)
    print(filename)
    index+=1




for epoch in range(Nepochs):  # loop over the dataset multiple times
    running_loss = 0.0
    totalLoss=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

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
        print('[epoch %d / %d, mini-batch %5d / %d] loss: %.3f' %(epoch + 1,Nepochs, i + 1,len(trainloader), running_loss))
        running_loss = 0.0
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
    f.write("{},{},{}\n".format(epoch+1,totalLoss,testLoss))
    f.close()
    #save the model
    if epoch%10==0:
        torch.save(model.state_dict(),'./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,epoch+1))


