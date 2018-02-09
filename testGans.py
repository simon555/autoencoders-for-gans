# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as pl

import visdom
vis=visdom.Visdom(server='http://eos11',port=24345,env='wass_gan') 



def rescale(img):
        mi=img.min()
        ma=img.max()
        output=((img-mi)/(ma-mi)-0)*1
        return(output)
        
transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Lambda(rescale)])
 
NumberOfEpochs=20000000
mb_size = 64
Z_dim = 100
X_dim = 28
Y_dim = 28
h_dim = 128
lr = 1e-3

trainset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=True,
                                            download=True, transform=transform)
        
mnist = torch.utils.data.DataLoader(trainset, batch_size=mb_size,
                                              shuffle=True, num_workers=0)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)



class Generator(nn.Module):
    def __init__(self,
                 useCuda=False):
        super(Generator, self).__init__()
        self.upSample= nn.Upsample(scale_factor=2, mode='bilinear')

        self.gen0=nn.Linear(Z_dim,15*15)    
        self.gen1=nn.Conv2d(1,8,3,padding=1)
        self.gen2=nn.Conv2d(8,16,3,padding=1)
        self.gen3=nn.Conv2d(16,32,3,padding=1)
        self.gen4=nn.Conv2d(32,16,3,padding=1)
        self.gen5=nn.Conv2d(16,8,3,padding=1)
        self.gen6=nn.Conv2d(8,1,3)   
        
        self.useCuda=torch.cuda.is_available()
        
        if self.useCuda:
            self.cuda()
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : generator')
        
    def forward(self,noise):
        x=F.relu(self.gen0(noise)).view(-1,1,15,15)
        x=F.relu(self.gen1(x))
        x=F.relu(self.gen2(x))
        x=F.relu(self.gen3(x))
        x = self.upSample(x)
        x=F.relu(self.gen4(x))
        x=F.relu(self.gen5(x))
        x = F.sigmoid(self.gen6(x))
        return (x)
    
G=Generator()

class Discriminator(nn.Module):
    def __init__(self,
                 useCuda=False):
        super(Discriminator, self).__init__()
        self.dis1=nn.Conv2d(1,8,3,stride=1)
        self.dis2=nn.Conv2d(8,16,3,stride=2)
        self.dis3=nn.Conv2d(16,32,3,stride=1)
        self.dis4=nn.Conv2d(32,16,3,stride=2)
        self.dis5=nn.Conv2d(16,8,3,stride=1)   
        self.dis6=nn.Linear(32,1)
        self.useCuda=torch.cuda.is_available()
        
        if self.useCuda:
            self.cuda()
        
        print('use CUDA : ',self.useCuda)        
        print('model loaded : discriminator')
        
    def forward(self,noise):
        x=F.relu(self.dis1(noise))
        #print(x.size())
        x=F.relu(self.dis2(x))
        #print(x.size())
        x=F.relu(self.dis3(x))
        #print(x.size())
        x=F.relu(self.dis4(x))
        #print(x.size())
        x=F.relu(self.dis5(x)).view(-1,32)
        x = self.dis6(x)
        return (x)

D=Discriminator()

# =============================================================================
# test
# =============================================================================
#noise=Variable(torch.randn(23,100))
#fake=G(noise)
#print('fake',fake.size())
#choice=D(fake)
#print('çhoice',choice.size())




G_solver = optim.Adam(G.parameters(), lr=5e-5)
D_solver = optim.Adam(D.parameters(), lr=5e-5)

ones_label=Variable(torch.ones(mb_size,1))
zeros_label=Variable(torch.zeros(mb_size,1))

if torch.cuda.is_available():
    ones_label=ones_label.cuda()
    zeros_label=zeros_label.cuda()



for epoch in range(NumberOfEpochs):
    
    for it in range(5):
        data=next(iter(mnist))
        # Sample data
        G_solver.zero_grad()
        D_solver.zero_grad()
        X, _ = data[0],data[1]
        X = Variable(X)
        
        
        tmp=X.size()[0]
        if mb_size!=tmp:
            mb_size=tmp
            ones_label=Variable(torch.ones(mb_size,1))
            zeros_label=Variable(torch.zeros(mb_size,1))
            if torch.cuda.is_available():
                ones_label=ones_label.cuda()
                zeros_label=zeros_label.cuda()


        z = Variable(torch.randn(mb_size, Z_dim))
        
        if torch.cuda.is_available():
            X=X.cuda()
            z=z.cuda()
    
    
        # Dicriminator forward-loss-backward-update
        # D(X) forward and loss
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)
    
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))   
        
        D_loss.backward()
        for p in D.parameters():
            
            p.grad.data.clamp_(-0.01,0.01)
        D_solver.step()
        
    
    
    # Generator forward-loss-backward-update
    ## some codes
      # Housekeeping - reset gradient
    G_solver.zero_grad()
    D_solver.zero_grad()
    
    
    
    
    
    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim))
    if torch.cuda.is_available():
        z=z.cuda()
        
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = -torch.mean(D_fake)    
    
    G_loss.backward()
    G_solver.step()
        
        
    
    if epoch%10==0:
        print(epoch)
        
        try:
            display=vis.images(G_sample.view(mb_size,1,28,28).cpu().data,
                    opts=dict(title='generated'),win=display)
        except:
            display=vis.images(G_sample.view(mb_size,1,28,28).cpu().data,
                    opts=dict(title='generated'))
            
        
       
