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
    def __init__(self, inputChannel,
                 outputChannel):        
        super(Block, self).__init__()
        
        self.Conv_1=nn.Conv2d(inputChannel,inputChannel,3,stride=1,padding=1) 
        self.BN_1=torch.nn.BatchNorm2d(inputChannel)
        self.Conv_2=nn.Conv2d(inputChannel,inputChannel,3,stride=1,padding=1)
        self.BN_2=torch.nn.BatchNorm2d(inputChannel)
        self.Conv_3=nn.Conv2d(inputChannel,outputChannel,3,stride=1,padding=1) 
        self.BN_3=torch.nn.BatchNorm2d(outputChannel)
        
       
            
    
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

        
        
        self.Code_B1=Block(3,32)
        self.Code_B1_1=Block(32,32)        
        self.Code_B2=Block(32,3)     
        
        self.Code_B3=Block(3,64)  
        self.Code_B3_1=Block(64,64) 
        self.Code_B4=Block(64,3)  
        
        self.Code_B5=Block(3,128)  
        self.Code_B5_1=Block(128,128) 
        self.Code_B6=Block(128,3)
        
        
        
        
        #decoding blocks
        self.DeCode_B1=Block(3,128) 
        self.DeCode_B1_1=Block(128,128) 
        self.DeCode_B2=Block(128,64)  
        #self.Conv_Decode1=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        
        self.DeCode_B3=Block(64,64)  
        self.DeCode_B3_1=Block(64,64)  
        self.DeCode_B4=Block(64,32)  
        
        #self.Conv_Decode2=nn.Conv2d(2*Nblocks,Nblocks,3,stride=1,padding=1) 
        self.DeCode_B5=Block(32,32)  
        self.DeCode_B5_1=Block(32,32)  
        self.DeCode_B6=Block(32,3)  
        
        self.DeCode_Conv1=nn.Conv2d(3,3,3,stride=1,padding=1)

                   
        if (self.useCuda):
            self.cuda()              
#            
#            #encoding blocks
#            self.Code_B1.cuda()
#            self.Code_B2.cuda()
#            self.Code_B3.cuda()
#            self.Code_B4.cuda()
#            self.Code_B5.cuda()
#            self.Code_B6.cuda()
#            
#            #decoding blocks
#            self.DeCode_B1.cuda()
#            self.DeCode_B2.cuda()
#            #self.Conv_Decode1.cuda()
#            self.DeCode_B3.cuda()
#            self.DeCode_B4.cuda()
#            #self.Conv_Decode2.cuda()
#            self.DeCode_B5.cuda()
#            self.DeCode_B6.cuda()
            
            
            
        print('use CUDA : ',self.useCuda)        
        print('model loaded : Resnet Modified')
        

 
    def code(self,image):             
        x=self.Code_B1.forward(image)
        x=self.Code_B1_1.forward(x)
        x=self.Code_B2.forward(x)       
        x=self.pool(x)           


        x=self.Code_B3.forward(x)
        x=self.Code_B3_1.forward(x)
        x=self.Code_B4.forward(x)
        x=self.pool(x)           

        

        #x=self.Code_Conv1(x)        
        x=self.Code_B5.forward(x)
        x=self.Code_B5_1.forward(x)
        x=self.Code_B6.forward(x)
        x=self.pool(x)           
        
        return(x)
        
        
    def decode(self,image):
         
               
        x=self.DeCode_B1.forward(image)
        x=self.DeCode_B1_1.forward(x)
        x=self.DeCode_B2.forward(x)   
        x=self.upSample(x)        

               
        
        x=self.DeCode_B3.forward(x)
        x=self.DeCode_B3_1.forward(x)
        x=self.DeCode_B4.forward(x)   
        x=self.upSample(x)        
              
        
        x=self.DeCode_B5.forward(x)
        x=self.DeCode_B5_1.forward(x)
        x=self.DeCode_B6.forward(x)
        x=self.upSample(x)        

        
        x=self.DeCode_Conv1(x)
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
    
