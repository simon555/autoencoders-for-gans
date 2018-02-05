import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var

class Discriminator_FC(nn.Module):

    def __init__(self,nin):
        super(Discriminator_FC, self).__init__()

        self.embed_z1 = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512,512))

        self.embed_z2 = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512,512))

        self.l1 = nn.Sequential(
            nn.Linear(nin, 512))

        self.l2 = nn.Sequential(
            nn.LeakyReLU(0.02),
            nn.Linear(512, 512))

        self.l3 = nn.Sequential(
            nn.LeakyReLU(0.02),
            nn.Linear(512, 1))

    def forward(self, x, z):

        ez1 = self.embed_z1(z)
        ez2 = self.embed_z2(z)

        out = self.l1(x) + ez1
        out = self.l2(out) + ez2
        out = self.l3(out)

        return out


class Discriminator_ConvDuck(nn.Module):
    def __init__(self, batch_size, nz):
        super(Discriminator_ConvDuck, self).__init__()
        self.batch_size = batch_size
        self.zo2 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256*6*8),
            nn.LeakyReLU(0.2))

        self.zo3 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512*3*4),
            nn.LeakyReLU(0.2))

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2))

        self.l_end = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=5, padding=2, stride=2))

    def forward(self, x, z):
        zo2 = self.zo2(z).view(self.batch_size,256,6,8) #goes to 256x8x8
        zo3 = self.zo3(z).view(self.batch_size,512,3,4)

        out = self.l1(x)
        out = self.l2(out)

        print out.size()
        print zo2.size()

        out = out + zo2
        out = self.l3(out)
        out = out + zo3

        out = self.l_end(out)
        out = out.view(self.batch_size,-1)
        return out


class Discriminator_ConvCifar(nn.Module):
    def __init__(self, batch_size, nz):
        super(Discriminator_ConvCifar, self).__init__()
        self.batch_size = batch_size
        self.zo2 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256*8*8),
            nn.LeakyReLU(0.2))
    
        self.zo3 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512*4*4),
            nn.LeakyReLU(0.2))

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2))
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2))

        self.l_end = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=5, padding=2, stride=2))

    def forward(self, x):
        #zo2 = self.zo2(z).view(self.batch_size,256,8,8) #goes to 256x8x8
        #zo3 = self.zo3(z).view(self.batch_size,512,4,4)

        out = self.l1(x)
        out = self.l2(out)
        out = out
        out = self.l3(out)
        out = out


        out = self.l_end(out)
        out = out.view(self.batch_size,-1)
        return out

    
class D_Bot_Conv32_old(nn.Module):
    def __init__(self, batch_size, nz):
        super(D_Bot_Conv32, self).__init__()
        self.batch_size = batch_size

        self.zo2 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256*8*8),
            nn.LeakyReLU(0.02))

        self.zo3 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 512*4*4),
            nn.LeakyReLU(0.02))

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.l_end = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=5, padding=2, stride=2))

    def forward(self, x,z):

        zo2 = self.zo2(z).view(self.batch_size,256,8,8) #goes to 256x8x8
        zo3 = self.zo3(z).view(self.batch_size,512,4,4)

        out = self.l1(x)

        print "out size l1", out.size()

        out = self.l2(out) + zo2
        out = self.l3(out) + zo3
        out = self.l_end(out)
        out = out.view(self.batch_size,-1)
        return out
    










