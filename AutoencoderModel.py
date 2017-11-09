

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

torch.manual_seed(1)


class Autoencoder(nn.Module):
    def __init__(self,
                 input_shape=(32,32,3)):
        
        super(Autoencoder, self).__init__()

       
        #visual parameters
        self.pool = nn.MaxPool2d(2, 2)
        self.input_shape=input_shape
        
        #first step
        self.Conv0=nn.Conv2d(3,4,3,stride=1,padding=1) 
        self.Conv1=nn.Conv2d(4,4,3,stride=1,padding=1) 
        
        #block1
        self.Conv_1B1=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_2B1=nn.Conv2d(4,4,1,stride=1,padding=0) 
        self.Conv_3B1=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_4B1=nn.Conv2d(8,4,3,stride=1,padding=1) 
        self.Conv_5B1=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_6B1=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_7B1=nn.Conv2d(8,4,3,stride=1,padding=1) 
        self.Conv_8B1=nn.Conv2d(3,4,3,stride=1,padding=1) 
        self.Conv_9B1=nn.Conv2d(8,4,3,stride=1,padding=1) 
        
        #block2
        self.Conv_1B2=nn.Conv2d(4,16,3,stride=1,padding=1) 
        self.Conv_2B2=nn.Conv2d(16,16,1,stride=1,padding=0) 
        self.Conv_3B2=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_4B2=nn.Conv2d(32,16,3,stride=1,padding=1) 
        self.Conv_5B2=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_6B2=nn.Conv2d(4,16,3,stride=1,padding=1) 
        self.Conv_7B2=nn.Conv2d(32,16,3,stride=1,padding=1) 
        self.Conv_8B2=nn.Conv2d(3,16,3,stride=1,padding=1) 
        self.Conv_9B2=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_10B2=nn.Conv2d(32,16,3,stride=1,padding=1) 
        
        #upBlock1
        self.Conv_1U1=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_2U1=nn.Conv2d(16,16,1,stride=1,padding=0) 
        self.Conv_3U1=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_4U1=nn.Conv2d(32,16,3,stride=1,padding=1)
        self.UP_1U1=nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.Conv_5U1=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.UP_2U1=nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.Conv_6U1=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_7U1=nn.Conv2d(32,16,3,stride=1,padding=1) 
        self.UP_3U1=nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.Conv_8U1=nn.Conv2d(16,16,3,stride=1,padding=1) 
        self.Conv_9U1=nn.Conv2d(32,4,3,stride=1,padding=1) 
        
        #upBlock2
        self.Conv_1U2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_2U2=nn.Conv2d(4,4,1,stride=1,padding=0) 
        self.Conv_3U2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_4U2=nn.Conv2d(8,4,3,stride=1,padding=1)
        self.UP_1U2=nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.Conv_5U2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.UP_2U2=nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.Conv_6U2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_7U2=nn.Conv2d(8,4,3,stride=1,padding=1) 
        self.UP_3U2=nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.Conv_8U2=nn.Conv2d(8,8,3,stride=1,padding=1) 
        self.UP_4U2=nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.Conv_9U2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv_10U2=nn.Conv2d(8,4,3,stride=1,padding=1) 
        
        #last step
        self.Conv2=nn.Conv2d(4,4,3,stride=1,padding=1) 
        self.Conv3=nn.Conv2d(4,3,3,stride=1,padding=1) 
        self.Conv4=nn.Conv2d(3,3,3,stride=1,padding=1) 
        self.Conv5=nn.Conv2d(3,3,3,stride=1,padding=1) 





        
        

 



    def forward(self,image):
        
# =============================================================================
#         CODE
# =============================================================================
        
        #first step
        x = self.Conv0(image)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        x = self.Conv1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        afterFirstStep = F.relu(x)



        
        #first block
        x = self.Conv_1B1(afterFirstStep)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        y1 = self.Conv_2B1(x)
        y1 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3B1(x)
        y2 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4B1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        x=self.pool(x)
        
        x = self.Conv_5B1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        z1 = F.relu(x)
        
        z2=self.pool(afterFirstStep)
        z2 = self.Conv_6B1(z2)
        z2 = torch.nn.BatchNorm2d(z2.size()[1],affine=False)(z2)
        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)

        x = self.Conv_7B1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        q1 = F.relu(x)
        
        
        q2=self.pool(image)
        q2 = self.Conv_8B1(q2)
        q2 = torch.nn.BatchNorm2d(q2.size()[1],affine=False)(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_9B1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        afterFirstBlock = F.relu(x)
        
        
        
        #second block
        x = self.Conv_1B2(afterFirstBlock)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        y1 = self.Conv_2B2(x)
        y1 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3B2(x)
        y2 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4B2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        x=self.pool(x)
        
        x = self.Conv_5B2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        z1 = F.relu(x)
        
        z2=self.pool(afterFirstBlock)
        z2 = self.Conv_6B2(z2)
        z2 = torch.nn.BatchNorm2d(z2.size()[1],affine=False)(z2)
        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)

        x = self.Conv_7B2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        q1 = F.relu(x)
        
        
        q2=self.pool(image)
        q2 = self.Conv_8B2(q2)
        q2=self.pool(q2)
        q2 = self.Conv_9B2(q2)
        q2 = torch.nn.BatchNorm2d(q2.size()[1],affine=False)(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_10B2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        code = F.relu(x)
        
        
# =============================================================================
#         DECODE
# =============================================================================
               
        #first UPblock
        x = self.Conv_1U1(code)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        y1 = self.Conv_2U1(x)
        y1 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3U1(x)
        y2 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4U1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        x=self.UP_1U1(x)
        
        x = self.Conv_5U1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        z1 = F.relu(x)
        
        z2=self.UP_2U1(code)
        z2 = self.Conv_6U1(z2)
        z2 = torch.nn.BatchNorm2d(z2.size()[1],affine=False)(z2)
        z2 = F.relu(z2)
        
        print("z1",z1.size())
        print("z2",z2.size())

        x=torch.cat((z1,z2),1)
        print("x",x.size())

        x = self.Conv_7U1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        q1 = F.relu(x)
        
        
        q2=self.UP_3U1(code)
        q2 = self.Conv_8U1(q2)
        q2 = torch.nn.BatchNorm2d(q2.size()[1],affine=False)(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_9U1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        afterFirstUp = F.relu(x)
        
        
        
        
        
        
        #   second UPblock
        x = self.Conv_1U2(afterFirstUp)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        y1 = self.Conv_2U2(x)
        y1 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y1)
        y1 = F.relu(y1)
        
        y2 = self.Conv_3U2(x)
        y2 = torch.nn.BatchNorm2d(x.size()[1],affine=False)(y2)
        y2 = F.relu(y2)
        
        x=torch.cat((y1,y2),1)
        
        x = self.Conv_4U2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        
        
        x=self.UP_1U2(x)
        
        x = self.Conv_5U2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        z1 = F.relu(x)
        
        z2=self.UP_2U2(afterFirstUp)
        z2 = self.Conv_6U2(z2)
        z2 = torch.nn.BatchNorm2d(z2.size()[1],affine=False)(z2)
        z2 = F.relu(z2)
        
        

        x=torch.cat((z1,z2),1)
        print("x",x.size())

        x = self.Conv_7U2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        q1 = F.relu(x)
        
        
        q2=self.UP_3U2(code)
        q2 = self.Conv_8U2(q2)
        q2=self.UP_4U2(q2)
        q2 = self.Conv_9U2(q2)
        q2 = torch.nn.BatchNorm2d(q2.size()[1],affine=False)(q2)
        q2 = F.relu(q2)
        
        x=torch.cat((q1,q2),1)

        x = self.Conv_10U2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
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

print(y.size())