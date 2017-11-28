

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

Nepochs=200000
NbatchTrain=200
NbatchTest=100

Nexperience=2



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
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

N1=16
N2=N1*N1

class Autoencoder(nn.Module):
    def __init__(self,
                 input_shape=(32,32,3)):
        
        super(Autoencoder, self).__init__()

       
        #visual parameters
        self.pool = nn.MaxPool2d(2, 2)
        self.input_shape=input_shape
        
        #first step

        self.Conv0=nn.Conv2d(3,N1,3,stride=1,padding=1)
        self.BN_1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv1=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_2=torch.nn.BatchNorm2d(N1,affine=False)

        #block1
        self.Conv_1B1=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_1B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_2B1=nn.Conv2d(N1,N1,1,stride=1,padding=0)
        self.BN_2B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_3B1=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_3B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_4B1=nn.Conv2d(2*N1,N1,3,stride=1,padding=1) 
        self.BN_4B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_5B1=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_5B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_6B1=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_6B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_7B1=nn.Conv2d(2*N1,N1,3,stride=1,padding=1) 
        self.BN_7B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_8B1=nn.Conv2d(3,N1,3,stride=1,padding=1) 
        self.BN_8B1=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_9B1=nn.Conv2d(2*N1,N1,3,stride=1,padding=1) 
        self.BN_9B1=torch.nn.BatchNorm2d(N1,affine=False)


        #block2
        self.Conv_1B2=nn.Conv2d(N1,N2,3,stride=1,padding=1) 
        self.BN_1B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_2B2=nn.Conv2d(N2,N2,1,stride=1,padding=0)
        self.BN_2B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_3B2=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_3B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_4B2=nn.Conv2d(2*N2,N2,3,stride=1,padding=1) 
        self.BN_4B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_5B2=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_5B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_6B2=nn.Conv2d(N1,N2,3,stride=1,padding=1) 
        self.BN_6B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_7B2=nn.Conv2d(2*N2,N2,3,stride=1,padding=1) 
        self.BN_7B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_8B2=nn.Conv2d(3,N2,3,stride=1,padding=1) 
        self.Conv_9B2=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_9B2=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_10B2=nn.Conv2d(2*N2,N2,3,stride=1,padding=1) 
        self.BN_10B2=torch.nn.BatchNorm2d(N2,affine=False)
      
        
        #upBlock1
        self.Conv_1U1=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_1U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_2U1=nn.Conv2d(N2,N2,1,stride=1,padding=0)
        self.BN_2U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_3U1=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_3U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_4U1=nn.Conv2d(2*N2,N2,3,stride=1,padding=1)
        self.BN_4U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.UP_1U1=nn.ConvTranspose2d(N2, N2, 2, stride=2)
        self.Conv_5U1=nn.Conv2d(N2,N2,3,stride=1,padding=1)
        self.BN_5U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.UP_2U1=nn.ConvTranspose2d(N2, N2, 2, stride=2)
        self.Conv_6U1=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_6U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_7U1=nn.Conv2d(2*N2,N2,3,stride=1,padding=1) 
        self.BN_7U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.UP_3U1=nn.ConvTranspose2d(N2, N2, 2, stride=2)
        self.Conv_8U1=nn.Conv2d(N2,N2,3,stride=1,padding=1) 
        self.BN_8U1=torch.nn.BatchNorm2d(N2,affine=False)
        self.Conv_9U1=nn.Conv2d(2*N2,N1,3,stride=1,padding=1) 
        self.BN_9U1=torch.nn.BatchNorm2d(N1,affine=False)
        
        #upBlock2
        self.Conv_1U2=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_1U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_2U2=nn.Conv2d(N1,N1,1,stride=1,padding=0) 
        self.BN_2U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_3U2=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_3U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_4U2=nn.Conv2d(2*N1,N1,3,stride=1,padding=1)
        self.BN_4U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.UP_1U2=nn.ConvTranspose2d(N1, N1, 2, stride=2)
        self.Conv_5U2=nn.Conv2d(N1,N1,3,stride=1,padding=1)
        self.BN_5U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.UP_2U2=nn.ConvTranspose2d(N1, N1, 2, stride=2)
        self.Conv_6U2=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_6U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_7U2=nn.Conv2d(2*N1,N1,3,stride=1,padding=1) 
        self.BN_7U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.UP_3U2=nn.ConvTranspose2d(N2, N2//2, 2, stride=2)
        self.Conv_8U2=nn.Conv2d(N2//2,N2//2,3,stride=1,padding=1) 
        #self.BN_8U2=torch.nn.BatchNorm2d(4,affine=False)
        self.UP_4U2=nn.ConvTranspose2d(N2//4, N1, 2, stride=2)
        self.Conv_9U2=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.BN_9U2=torch.nn.BatchNorm2d(N1,affine=False)
        self.Conv_10U2=nn.Conv2d(2*N1,N1,3,stride=1,padding=1) 
        self.BN_10U2=torch.nn.BatchNorm2d(N1,affine=False)
        
        #last step
        self.Conv2=nn.Conv2d(N1,N1,3,stride=1,padding=1) 
        self.Conv3=nn.Conv2d(N1,3,3,stride=1,padding=1) 
        self.Conv4=nn.Conv2d(3,3,3,stride=1,padding=1) 
        self.Conv5=nn.Conv2d(3,3,3,stride=1,padding=1) 





        
        

 



    def forward(self,image):
        
# =============================================================================
#         CODE
# =============================================================================
        
        #first step
        x = self.Conv0(image)
        x = self.BN_1(x)
        x = F.relu(x)
        
        x = self.Conv1(x)
        x = self.BN_2(x)
        afterFirstStep = F.relu(x)



        
        #first block
        x = self.Conv_1B1(afterFirstStep)
        x = self.BN_1B1(x)
        x = F.relu(x)
        
        y1 = self.Conv_2B1(x)
        y1 = self.BN_2B1(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3B1(x)
        y2 = self.BN_3B1(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4B1(x)
        x = self.BN_4B1(x)
         
        x = F.relu(x)
        
        x=self.pool(x)
        
        x = self.Conv_5B1(x)
        x = self.BN_5B1(x)
        z1 = F.relu(x)
        
        z2=self.pool(afterFirstStep)
        z2=self.Conv_6B1(z2)
        z2= self.BN_6B1(z2)
        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)

        x = self.Conv_7B1(x)
        x = self.BN_7B1(x)
        q1 = F.relu(x)
        
        
        q2=self.pool(image)
        q2 = self.Conv_8B1(q2)
        q2 = self.BN_8B1(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_9B1(x)
        x = self.BN_9B1(x)
        afterFirstBlock = F.relu(x)
        
        
        
        #second block
        x = self.Conv_1B2(afterFirstBlock)
        x = self.BN_1B2(x)
        x = F.relu(x)
        
        y1 = self.Conv_2B2(x)
        y1 = self.BN_2B2(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3B2(x)
        y2 = self.BN_3B2(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4B2(x)
        x = self.BN_4B2(x)

        x = F.relu(x)
        
        x=self.pool(x)        
        x = self.Conv_5B2(x)
        x = self.BN_5B2(x)

        z1 = F.relu(x)
        
        z2=self.pool(afterFirstBlock)
        z2 = self.Conv_6B2(z2)
        z2 = self.BN_6B2(z2)

        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)

        x = self.Conv_7B2(x)
        x = self.BN_7B2(x)

        q1 = F.relu(x)
        
        
        q2=self.pool(image)
        q2 = self.Conv_8B2(q2)
        q2=self.pool(q2)
        q2 = self.Conv_9B2(q2)
        q2= self.BN_9B2(q2)

        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_10B2(x)
        x = self.BN_10B2(x)
        code = F.relu(x)
        
        
# =============================================================================
#         DECODE
# =============================================================================
               
        #first UPblock
        x = self.Conv_1U1(code)
        x = self.BN_1U1(x)
        x = F.relu(x)
        
        y1 = self.Conv_2U1(x)
        y1= self.BN_2U1(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3U1(x)
        y2= self.BN_3U1(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4U1(x)
        x= self.BN_4U1(x)

        x = F.relu(x)
        x=self.UP_1U1(x)
        
        x = self.Conv_5U1(x)
        x= self.BN_5U1(x)

        z1 = F.relu(x)
        
        z2=self.UP_2U1(code)
        z2 = self.Conv_6U1(z2)
        z2=self.BN_6U1(z2)
        z2 = F.relu(z2)      
        
       

        x=torch.cat((z1,z2),1)
        

        x = self.Conv_7U1(x)
        x= self.BN_7U1(x)
        q1 = F.relu(x)
        
        
        q2=self.UP_3U1(code)
        q2 = self.Conv_8U1(q2)
        q2= self.BN_8U1(q2)

        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_9U1(x)
        x= self.BN_9U1(x)

        afterFirstUp = F.relu(x)
        
        
        
        
        
        
        #   second UPblock
        x = self.Conv_1U2(afterFirstUp)
        x= self.BN_1U2(x)
        x = F.relu(x)
        
        y1 = self.Conv_2U2(x)
        y1= self.BN_2U2(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3U2(x)
        y2= self.BN_3U2(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4U2(x)
        x= self.BN_4U2(x)
        x = F.relu(x)
        
        
        x=self.UP_1U2(x)
        
        x = self.Conv_5U2(x)
        x= self.BN_5U2(x)
        z1 = F.relu(x)
        
        z2=self.UP_2U2(afterFirstUp)
        z2 = self.Conv_6U2(z2)
        z2= self.BN_6U2(z2)
        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)
        x = self.Conv_7U2(x)
        x= self.BN_7U2(x)
        q1 = F.relu(x)
        
        
        q2=self.UP_3U2(code)
        q2 = self.Conv_8U2(q2)
        q2=self.UP_4U2(q2)
        q2 = self.Conv_9U2(q2)
        q2= self.BN_9U2(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_10U2(x)
        x= self.BN_10U2(x)
        afterSecondUp = F.relu(x)
        
        
 
        
# =============================================================================
#         Last step
# =============================================================================
        
        x=self.Conv2(afterSecondUp)
        x = F.relu(x)
        x=self.Conv3(x)
        x = F.relu(x)
        x=self.Conv4(x)
        x = F.relu(x)
        x=self.Conv5(x)
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
optimizer=optim.Adam(model.parameters(), lr=0.001)
#optimizer=optim.Adadelta(model.parameters())
filename="./results/Exp{}/data/data.txt".format(Nexperience)
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
    if epoch%1000==0:
        torch.save(model,'./results/Exp{}/models/Exp{}Epoch{}.pt'.format(Nexperience,Nexperience,epoch+1))


