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
    
def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class myDownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(myDownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.pooling:
            x = self.pool(x)
        return (x)

class myUpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(myUpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.out_channels, self.out_channels, 
            mode=self.up_mode)
        
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        
        
        
    def forward(self, from_down, from_left):
        """ Forward pass
        Arguments:
            from_down: tensor from the down-bock
            from_left: tensor from the left-block
        """
       
        
       
        if self.merge_mode == 'concat':
            x = torch.cat((from_down, from_left), 1)
        else:
            x = from_down + from_down

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upconv(x))
        
        return x

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
       
        from_up = self.upconv(from_up)
        
       
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ModelAE(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, output_channels=3, in_channels=3, depth=2, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat',lastActivation='sigmoid'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(ModelAE, self).__init__()
        
        self.useCuda=torch.cuda.is_available()
        if self.useCuda:
            self.cuda()

        self.depth=depth
        self.lastActivation=lastActivation
        self.encoder=Encoder(depth=self.depth)
        self.decoder=Decoder(depth=self.depth,lastActivation=self.lastActivation)
        print('model loaded : GridNet Modified')

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
    
    def forward(self, x):
        return (self.decoder(self.encoder(x)))
    
class Encoder(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, output_channels=512, in_channels=3, depth=2, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(Encoder, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.output_channels = start_filts*(2**(depth))
        print('output channels ',output_channels)
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.shapeOfCode=self.start_filts*(2**(self.depth-1))
        

        self.down_convs = []
        self.up_convs = [[]]



       
        # create the encoder pathway and add to a list
        for i in range(self.depth):
            self.down_convs.append([])
            for j in range(i,self.depth):
                if i==0 and j==0:
                    ins = self.in_channels
                    outs=self.start_filts
                    pooling=False
                else:
                    ins = self.start_filts*(2**(j-1))
                    outs=2*ins
                    pooling = True #if i < depth-1 else False

                #outs= self.start_filts*(2**(j+1))
                print('i : {}, j : {}, ins : {}, out : {}'.format(i,j,ins,outs))
                down_conv = myDownConv(ins, outs, pooling=pooling)
                self.down_convs[i].append(down_conv)
            #self.test= nn.ModuleList(self.down_convs[i])
            self.down_convs[i] = nn.ModuleList(self.down_convs[i])
            
        self.down_convs= nn.ModuleList(self.down_convs)
        #print(len(self.down_convs))
        #print(self.down_convs[0])
            

        
        self.reset_params()
        
        self.useCuda=torch.cuda.is_available()
        if self.useCuda:
            self.cuda()
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : Inception Modified')

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)



        
        
        
        
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
    
   
    
    
    def forward(self, inp):
        
         
        # encoder pathway, save outputs for merging
        for i, moduleList in enumerate(self.down_convs):
            for l,module in enumerate(moduleList): 
                if l==0:
                    inp=module(inp)
                    x=inp
                else:
                    x=module(x)
            if i ==0:
                output=x
            else:
                output=torch.cat([output,x],1)      
           
            
        
        return (output)

class Decoder(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, output_channels=3, in_channels=2048, depth=2, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat',lastActivation='sigmoid'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(Decoder, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.output_channels = output_channels
        print('output channels ',output_channels)
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.lastActivation=lastActivation
        self.shapeOfCode=self.start_filts*(2**(self.depth-1))
        

        self.up_convs = []

                

       
        # create the encoder pathway and add to a list
        for i in range(self.depth-1):
            self.up_convs.append([])
            for j in range(self.depth-i-1):
                ins = self.shapeOfCode//(2**(j))
                outs=ins//2

                #outs= self.start_filts*(2**(j+1))
                up_conv = myUpConv(ins, outs)
                self.up_convs[i].append(up_conv)
                print('i : {}, j : {}, ins : {}, out : {}'.format(i,j,ins,outs))

            #self.test= nn.ModuleList(self.down_convs[i])
            if i ==0:
                finalConv=conv1x1(outs,self.output_channels)
                self.up_convs[i].append(finalConv)
                print('i : {}, j : {}, ins : {}, out : {}'.format(i,j+1,outs,self.output_channels))

            self.up_convs[i] = nn.ModuleList(self.up_convs[i])

                
        self.up_convs= nn.ModuleList(self.up_convs)
        #print(len(self.down_convs))
        #print(self.down_convs[0])
            

        
        self.reset_params()
        
        self.useCuda=torch.cuda.is_available()
        if self.useCuda:
            self.cuda()
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : Inception Modified')

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
        
        
    def getBlock(self, i,bigCode,shapeOfCode):
        begin=shapeOfCode*i
        end=shapeOfCode*(i+1)
        
        return(bigCode[:,begin:end,:,:])
        

   
    
    
    def forward(self, inp):
        
        
        # encoder pathway, save outputs for merging
        for j in range(self.depth-1):
            for i in range(self.depth-1-j):
                shapeOfCode=self.shapeOfCode//(2**(j))
                #print('expected shape of code : ', shapeOfCode)
                input1=self.getBlock(i,inp,shapeOfCode)
                input2=self.getBlock(i+1,inp,shapeOfCode)
                temp=self.up_convs[i][j](input1,input2)
                
                if i==0:
                    tempCode=temp
                else:
                    tempCode=torch.cat([tempCode,temp],1)
            inp=tempCode
            #print('new input size : ',inp.size())
            
        output=self.up_convs[0][self.depth-1](inp)
                
        if self.lastActivation=='relu':
            output=F.relu(output)
        elif self.lastActivation=='sigmoid':
            output=F.sigmoid(output)
            
        
        return (output)
    

class DecoderDouble(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, output_channels=3, in_channels=2048, depth=2, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat',lastActivation='sigmoid'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(DecoderDouble, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.output_channels = output_channels
        print('output channels ',output_channels)
        self.start_filts = start_filts
        self.depth = depth
        self.lastActivation=lastActivation
        self.shapeOfCode=self.start_filts*(2**(self.depth-1))
        self.in_channels = self.shapeOfCode*self.depth


        self.up_convs = []

    
        self.adapt1=conv3x3(self.in_channels,self.in_channels//2)
        self.adapt2=conv3x3(self.in_channels,self.in_channels//2)
        self.adapt3=conv3x3(self.in_channels,self.in_channels)


        
        
        # create the encoder pathway and add to a list
        for i in range(self.depth-1):
            self.up_convs.append([])
            for j in range(self.depth-i-1):
                ins = self.shapeOfCode//(2**(j))
                outs=ins//2

                #outs= self.start_filts*(2**(j+1))
                up_conv = myUpConv(ins, outs)
                self.up_convs[i].append(up_conv)
                print('i : {}, j : {}, ins : {}, out : {}'.format(i,j,ins,outs))

            #self.test= nn.ModuleList(self.down_convs[i])
            if i ==0:
                finalConv=conv1x1(outs,self.output_channels)
                self.up_convs[i].append(finalConv)
                print('i : {}, j : {}, ins : {}, out : {}'.format(i,j+1,outs,self.output_channels))

            self.up_convs[i] = nn.ModuleList(self.up_convs[i])

                
        self.up_convs= nn.ModuleList(self.up_convs)
        #print(len(self.down_convs))
        #print(self.down_convs[0])
            

        
        self.reset_params()
        
        self.useCuda=torch.cuda.is_available()
        if self.useCuda:
            self.cuda()
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : Inception Modified')

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
        
        
    def getBlock(self, i,bigCode,shapeOfCode):
        begin=shapeOfCode*i
        end=shapeOfCode*(i+1)
        
        return(bigCode[:,begin:end,:,:])
        

   
    
    
    def forward(self, inp1,inp2):
        inp1=self.adapt1(inp1)
        inp2=self.adapt2(inp2)
        inp=torch.cat([inp1,inp2],dim=1)
        inp=self.adapt3(inp)
        
        
        # encoder pathway, save outputs for merging
        for j in range(self.depth-1):
            for i in range(self.depth-1-j):
                shapeOfCode=self.shapeOfCode//(2**(j))
                #print('expected shape of code : ', shapeOfCode)
                input1=self.getBlock(i,inp,shapeOfCode)
                input2=self.getBlock(i+1,inp,shapeOfCode)
                temp=self.up_convs[i][j](input1,input2)
                
                if i==0:
                    tempCode=temp
                else:
                    tempCode=torch.cat([tempCode,temp],1)
            inp=tempCode
            #print('new input size : ',inp.size())
            
        output=self.up_convs[0][self.depth-1](inp)
                
        if self.lastActivation=='relu':
            output=F.relu(output)
        elif self.lastActivation=='sigmoid':
            output=F.sigmoid(output)
            
        
        return (output)
    


class Model(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self):
        super(Model, self).__init__()

        self.encoderA=Encoder()
        self.encoderA_INTER_B=Encoder()
        self.encoderB=Encoder()
        
        self.decoderA=DecoderDouble()
        self.decoderB=DecoderDouble()
        
        self.auxEncoderA=Encoder()
        self.auxDecoderA=Decoder()
        
        self.auxEncoderB=Encoder()
        self.auxDecoderB=Decoder()
        
        self.auxEncoderA_UNION_B=Encoder()
        self.auxDecoderA_UNION_B=Decoder()
        
        self.useCuda=torch.cuda.is_available()
        
      

    def forward(self, A, B,both):
        
        
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
        
        auxCodeA_UNION_B=self.auxEncoderA_UNION_B(both)
        auxReconstrutionA_UNION_B=self.auxDecoderA_UNION_B(auxCodeA_UNION_B)
        
        
        return (codeA,codeA_INTER_B_fromA,reconstructionA,
                codeB,codeA_INTER_B_fromB,reconstructionB,
                auxCodeA,auxReconstructionA,
                auxCodeB,auxReconstructionB,
                auxCodeA_UNION_B,auxReconstrutionA_UNION_B)

    


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