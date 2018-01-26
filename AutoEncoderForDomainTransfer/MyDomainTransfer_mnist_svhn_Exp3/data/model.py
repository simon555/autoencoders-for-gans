import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as pl

import sys
import os

currentDirectory = os.getcwd()
if not currentDirectory in sys.path:
    print('adding local directory : ', currentDirectory)
    sys.path.insert(0,currentDirectory)
    
import MyDeep as modelFactory



class Model(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self):
        super(Model, self).__init__()

        self.encoderA=modelFactory.Encoder()
        self.encoderA_INTER_B=modelFactory.Encoder()
        self.encoderB=modelFactory.Encoder()
        
        self.decoderA=modelFactory.DecoderDouble()
        self.decoderB=modelFactory.DecoderDouble()
        
        self.auxEncoderA=modelFactory.Encoder()
        self.auxDecoderA=modelFactory.Decoder()
        
        self.auxEncoderB=modelFactory.Encoder()
        self.auxDecoderB=modelFactory.Decoder()
        
        self.useCuda=torch.cuda.is_available()
        
      

    def forward(self, A, B):
        
        
        codeA=self.encoderA(A)        
        codeA_INTER_B_fromA=self.encoderA_INTER_B(A)        
        reconstructionA=self.decoderA(codeA,codeA_INTER_B_fromA)
        
        codeB=self.encoderB(B)        
        codeA_INTER_B_fromB=self.encoderA_INTER_B(B)        
        reconstructionB=self.decoderB(codeB,codeA_INTER_B_fromB)
        
        auxCodeA=self.auxEncoderA(A)
        auxReconstructionA=self.auxDecoderA(auxCodeA)
        
        auxCodeB=self.auxEncoderB(B)
        auxReconstructionB=self.auxDecoderB(auxCodeB)
        
        
        
        return (codeA,codeA_INTER_B_fromA,reconstructionA,
                codeB,codeA_INTER_B_fromB,reconstructionB,
                auxCodeA,auxReconstructionA,
                auxCodeB,auxReconstructionB)

    


if __name__ == "__main__":
    """
    testing
    """
    print('testing model domain transfer')
    
    model = Model()

    if model.useCuda:
        A= Variable(torch.FloatTensor(np.random.random((2, 3, 32, 32))).cuda())
        B= Variable(torch.FloatTensor(np.random.random((2, 3, 32, 32))).cuda())
        
    
    else:
        print('no CUDA!')        
        
        
    out = model(A,B)
    for u in out:
        print(u.size())
#    
#    
#    #loss = torch.sum(out)
#    #loss.backward()
#
#
#    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
#    print('number of parameters : ',params)
#    
#    
#    optimizer=optim.Adam(model.parameters(), lr=0.001)
#    criterion=torch.nn.MSELoss().cuda()    
#
#    running_loss = []
#
#    for epoch in range(10):  # loop over the dataset multiple times
#    
#        # get the inputs
#        #print('shape ', inputs.size()) 
#        # wrap them in Variable
#
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(x)
#        loss = criterion(outputs, testOutput)
#        loss.backward()
#        optimizer.step()
#    # print statistics
#        running_loss+=[loss.data[0]]
#        print('epoch {}   loss {} '.format(epoch,loss.data[0]))
#        
#    pl.plot(running_loss)

#    y=model.encode(x)
#    print('main code shape ', y.size())
#    
#    
#    u=model.decode(y)
#    print('output of decoder ', u,size())
#    LE=model.encoder_outs
#    LD=model.decoder_outs
#    print('original shape')
#    print(x.size())
#    print('encoder intermediate shapes')
#    for temp in LE:
#        print(temp.size())
#    print('decoder intermediate shapes')
#    for temp in LD:
#        print(temp.size())
#    print('final shape')
#    print(out.size())