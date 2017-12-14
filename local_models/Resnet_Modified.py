# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:23:42 2017

@author: simon
"""


import torch
import matplotlib.pyplot as pl
import torch.nn as nn
import torch.nn.functional as F



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

      



class ModelAE(nn.Module):
    def __init__(self,
                 useCuda=False,
                 inputChannel=3):
        
        super(ModelAE, self).__init__()


        useCuda=torch.cuda.is_available()
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
#            
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
        x = F.sigmoid(x)
        x=self.Conv_2F(x)
        x = F.sigmoid(x)
        x=self.Conv_3F(x)
        x = F.sigmoid(x)
        x=self.Conv_4F(x)
        x = F.sigmoid(x)   

        
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
    
