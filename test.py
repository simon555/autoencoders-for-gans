

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



print("modules loaded ")
torch.manual_seed(1)

Nepochs=40
NbatchTrain=32
NbatchTest=100
Nplot=1
Nsave=10
Nexperience=10



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

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
                 input_shape=(28,28,1)):
        
        super(Autoencoder, self).__init__()

       
        #visual parameters
        self.pool = nn.MaxPool2d(2, 2)
        self.upSample= nn.Upsample(scale_factor=2, mode='bilinear')
        self.input_shape=input_shape
        
        N1=32
        
        Nblocks=64
        
        #first step
        self.inputChannel=1
        self.Conv_1P=nn.Conv2d(self.inputChannel,N1,3,stride=1,padding=1)
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
        self.Conv_2F=nn.Conv2d(N1,self.inputChannel,3,stride=1,padding=1) 
        self.Conv_3F=nn.Conv2d(self.inputChannel,self.inputChannel,3,stride=1,padding=1) 
        self.Conv_4F=nn.Conv2d(self.inputChannel,self.inputChannel,3,stride=1,padding=1)       
        

 
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
        x1=self.Code_B2.forward(x)
        
        x=self.pool(x1)           
        x=self.Code_B3.forward(x)
        x2=self.Code_B4.forward(x)
        
        x=self.pool(x2)        
        x=self.Code_B5.forward(x)
        x=self.Code_B6.forward(x)

        
        return(x)
        
    def decode(self,image):
         
        x=self.Conv_1F(image)
        x = F.relu(x)
        x=self.Conv_2F(x)
        x = F.relu(x)
        x=self.Conv_3F(x)
        x = F.relu(x)
        x=self.Conv_4F(x)
        x = F.relu(x)   
        
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

        
        return(x)
        



    def forward(self,image):
        
# =============================================================================
#         Pre Step
# =============================================================================
        
        c=self.code(image)
        d=self.decode(c)
             
        
        return(d)
def plot_test(inputs,outputs):
    fig=pl.figure()
    pl.subplot(131)
    pl.imshow(inputs.cpu().data.numpy()[0,0,:,:],cmap='gray')
    pl.subplot(132)
    pl.imshow(outputs.cpu().data.numpy()[0,0,:,:],cmap='gray')
    pl.show()



model=Autoencoder()
print('model loaded')

model.cuda()
print('cuda loaded')
print('what ???')
criterion=torch.nn.MSELoss().cuda()
print('critetion loaded')

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer=optim.Adam(model.parameters(), lr=learningRate)
print('optimizer loaded')

#optimizer=optim.Adadelta(model.parameters())
directory='./results/Exp{}/'.format(Nexperience)
directoryData=directory+'data/'
directoryModel=directory+'models/'
if not os.path.exists(directory):
    print('new directory for : ',directory)
    os.makedirs(directory)    
    os.makedirs(directoryData)
    os.makedirs(directoryModel)
else:
    print('directory already exists : ',directory)



filename="./results/Exp{}/data/data_lr-{}.txt".format(Nexperience,learningRate)
index=2
while(os.path.exists(filename)):
    print("file aldready existing, using a new path ",end=" ")
    filename="./results/Exp{}/data/data-{}.txt".format(Nexperience,index)
    print(filename)
    index+=1
print('saving results at : ',filename)



print('beginning of the training')
for epoch in range(Nepochs):  # loop over the dataset multiple times
    running_loss = 0.0
    totalLoss=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        print('shape ', inputs.size()) 
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
    f.write("{},{},{}\n".format(epoch+1,totalLoss,testLoss))
    f.close()
    
    #save the model
    if epoch%Nsave==0:
        torch.save(model.state_dict(),'./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,epoch+1))


