import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var


class Generator_FC(nn.Module):

    def __init__(self,nout):
        super(Generator_FC, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(64, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, nout),
            nn.Tanh())

    def forward(self, z):
        out = self.l1(z)
        return out

class Generator_ConvCifar(nn.Module):
    def __init__(self, batch_size, nz, num_steps_total=1):
        super(Generator_ConvCifar, self).__init__()
        self.batch_size = batch_size
        self.nz = nz
        self.l1 = nn.Sequential(
            nn.Linear(nz, 512*4*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2, stride=1))

        self.bn1 = nn.BatchNorm2d(256,1)

        self.l3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1))

        self.bn2 = nn.BatchNorm2d(128)

        self.l4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1))

        self.bn3 = nn.BatchNorm2d(128)

        self.l5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1))

        self.bn4 = nn.BatchNorm2d(64)

        self.l6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=5, padding=2, stride=1),
            nn.Tanh())

    def forward(self, z, step=0):
        print "no extra noise in decoder"
        out = self.l1(z)
        out = out.view(self.batch_size,512,4,4)
        out = self.l2(out)
        h2 = self.bn1(out)
        out = self.l3(h2)
        h3 = self.bn2(out)
        h4l = self.l4(h3)
        h4 = self.bn3(h4l)
        h5l = self.l5(h4)
        out = self.bn4(h5l)
        out = self.l6(out)
        
        print "gen size", out.size()
        
        return out, h4l

#Returns 96x128
#3x4 -> 6x8 -> 12x16 -> 24x32 -> 48x64 -> 96x128

class Generator_ConvDuck(nn.Module):
    def __init__(self, batch_size, nz):
        super(Generator_ConvDuck, self).__init__()
        self.batch_size = batch_size
        self.nz = nz
        self.l1 = nn.Sequential(
            nn.Linear(nz*2, 512*3*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256,affine=True),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64,affine=True),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32,affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=5, padding=2, stride=1),
            nn.Tanh())
    def forward(self, z):
        print "no extra noise in decoder"
        z_extra = 0.0 * to_var(torch.randn(self.batch_size, self.nz))
        out = self.l1(torch.cat((z,z_extra), 1))
        out = out.view(self.batch_size,512,3,4)
        out = self.l2(out)
        return out




class Gen_Bot_Conv32(nn.Module):
    def __init__(self, batch_size,nz):
        super(Gen_Bot_Conv32, self).__init__()
        self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 512*4*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=5, padding=2, stride=1),
            nn.Tanh())
    def forward(self, z, give_pre=False):
        if give_pre:
            out = z
        else:
            out = self.l1(z)
            out = out.view(self.batch_size,512,4,4)
        out = self.l2(out)
        return out


class Gen_Bot_Conv32_deep1(nn.Module):
    def __init__(self, batch_size,nz):
        super(Gen_Bot_Conv32_deep1, self).__init__()
        self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 512*4*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),
            nn.Tanh())
    def forward(self, z, give_pre=False):
        if give_pre:
            out = z
        else:
            out = self.l1(z)
            out = out.view(self.batch_size,512,4,4)
        out = self.l2(out)
        return out

class Gen_Bot_Conv32_deepbottleneck(nn.Module):
    def __init__(self, batch_size,nz):
        super(Gen_Bot_Conv32_deepbottleneck, self).__init__()
        self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 32*4*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1),
            nn.Tanh())
    def forward(self, z, give_pre=False):
        if give_pre:
            out = z
        else:
            out = self.l1(z)
            out = out.view(self.batch_size,32,4,4)
        out = self.l2(out)
        return out



class Gen_Bot_Joint(nn.Module):
    def __init__(self, batch_size,nz):
        super(Gen_Bot_Joint, self).__init__()
        self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 32*4*4))
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 3*5, kernel_size=1, padding=0, stride=1),
            nn.Tanh())
    def forward(self, z):
        out = self.l1(z)
        out = out.view(self.batch_size,32,4,4)
        out = self.l2(out)
        return out




