
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo



class ModelAE(nn.Module):
    def __init__(self,
                 useCuda=False,
                 inputChannels=3,
                 depth=4):
        super(ModelAE, self).__init__()
               
        self.useCuda=torch.cuda.is_available()
        self.inputChannels=inputChannels
        self.depth=depth
        
        #define the convolutions for the encoder
        self.EncodeConvs=[]
        for i in range(self.depth):
            EncodeLayer=nn.Conv2d(self.inputChannels,self.inputChannels,2,stride=2,padding=0) 
            self.EncodeConvs.append(EncodeLayer)
        self.EncodeConvs = nn.ModuleList(self.EncodeConvs)

        #define the convolutions for the decoder
        self.DeconvUpsample=[]
        self.DeconvConvs=[]
        for i in range(self.depth):
            decodeLayer=nn.Conv2d(self.inputChannels,self.inputChannels,3,stride=1,padding=1) 
            decodeUpsample=nn.ConvTranspose2d(3,3,kernel_size=2,stride=2)
            self.DeconvConvs.append(decodeLayer)
            self.DeconvUpsample.append(decodeUpsample)
        
        
        self.DeconvConvs = nn.ModuleList(self.DeconvConvs)
        self.DeconvUpsample = nn.ModuleList(self.DeconvUpsample)

        self.lastConv=nn.Conv2d(self.inputChannels,self.inputChannels,1,stride=1,padding=0) 

        
        if self.useCuda:
            self.cuda()
        
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : plain')
        
    def forward(self,image):
        for i, module in enumerate(self.EncodeConvs):
            image=module(image)
            image=F.relu(image)
            #print(image.size())
            
        for i, module in enumerate(self.DeconvConvs):
            image=module(image)
            image=F.relu(image)
            deconvModule=self.DeconvUpsample[i]
            image=deconvModule(image)
            #print(image.size())
#        
        image=self.lastConv(image)
        image=F.sigmoid(image)
        return(image)
        
        
#x=Variable(torch.Tensor(1,3,32,32))
#model=ModelAE()
#y=model(x)
#print(y.size())
#import numpy as np
#
#
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print('parameters', params)
#        

