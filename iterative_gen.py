'''
Code for generating starting from a looser bottleneck to final images.  


'''

import sys
import os

use_different_pytorch = True

if use_different_pytorch:
    sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')

currentDirectory = os.getcwd()
if not currentDirectory in sys.path:
    print('adding local directory : ', currentDirectory)
    sys.path.insert(0,currentDirectory)

import featureGenerator.generateFeatures as generateFeatures
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from Generators import Generator_ConvCifar
from Discriminators import Discriminator_ConvCifar
from reg_loss import gan_loss

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def rescale(img):
    mi=img.min()
    ma=img.max()
    return(((img-mi)/(ma-mi)-0)*1)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

NbatchTrain=128
use_penalty = True

G = Generator_ConvCifar(NbatchTrain,64)
D = Discriminator_ConvCifar(NbatchTrain,64)

G = G.cuda()
D = D.cuda()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.0,0.99))
g_optimizer = torch.optim.Adam(list(G.parameters()), lr=0.0001, betas=(0.0,0.99))

#transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(rescale)])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])


model = generateFeatures.ModelAE()

trainset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='train',
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=NbatchTrain,
                                              shuffle=True, num_workers=0)

index = Variable(torch.from_numpy(np.array(1)))

for i, data in enumerate(trainloader, 0):
    image, label = data
    image = to_var(image)
    #h0 = model.getFeatureAtStep(0,image)
    h_128_16_16_real = model.getFeatureAtStep(1,image)
    #h2 = model.getFeatureAtStep(2,image)
    #h3 = model.getFeatureAtStep(3,image)
    #h4 = model.getFeatureAtStep(4,image)

    print "size of real get features", h_128_16_16_real.size()

    z = to_var(torch.randn(NbatchTrain, 64))

    genx,h_128_16_16_gen = G(z)

    D_gen = D(genx,h_128_16_16_gen*0.0)
    D_real = D(image,h_128_16_16_real*0.0)

    d_loss_gen = gan_loss(pre_sig=D_gen, real=False, D=True, use_penalty=use_penalty,grad_inp=genx,gamma=0.1) + gan_loss(pre_sig=D_gen, real=False, D=True, use_penalty=use_penalty,grad_inp=h_128_16_16_gen,gamma=0.1)

    D.zero_grad()
    d_loss_gen.backward(retain_graph=True)
    d_optimizer.step()

    d_loss_real = gan_loss(pre_sig=D_real, real=True, D=True, use_penalty=use_penalty,grad_inp=image,gamma=0.1) + gan_loss(pre_sig=D_real, real=True, D=True, use_penalty=use_penalty,grad_inp=h_128_16_16_real,gamma=0.1)
    D.zero_grad()
    d_loss_real.backward(retain_graph=True)
    d_optimizer.step()

    print d_loss_real + d_loss_gen
    
    g_loss_gen = gan_loss(pre_sig=D_gen, real=False, D=False, use_penalty=False,grad_inp=None,gamma=1.0)
    G.zero_grad()
    g_loss_gen.backward()
    g_optimizer.step()

    #print "h0", h0.size()
    #print "h1", h1.size()
    #print "h2", h2.size()
    #print "h3", h3.size()
    #print "h4", h4.size()

    if i % 5 == 0:
        print "saving images!"
        save_image(denorm(genx.data), 'plots/image_gen.png')
        save_image(denorm(image.data), 'plots/image_real.png')




