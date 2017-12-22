# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:41:22 2017

@author: simon
"""

import numpy
import matplotlib.pyplot as pl

Nexp=13

path="C://Users//simon//Desktop//MILA//autoencoders-for-gans//results//Exp{}//data//data.txt".format(Nexp)
#retraining
#path="C://Users//simon//Desktop//MILA//autoencoders-for-gans//results//Exp8//retrainFromEpoch281-2//data//data.txt"


f=open(path,'r')
trainList=[]
testList=[]
epoch=[]
for i,line in enumerate(f.readlines()):  
    if i>9:            
        idx,train,test=line.split(',')
        epoch+=[float(idx)]
        trainList+=[float(train)]
        testList+=[float(test)]

pl.plot(epoch,trainList,label='training set')
pl.plot(epoch,testList,label='test set')
pl.legend()
pl.savefig('C://Users//simon//Desktop//MILA//autoencoders-for-gans//results//Exp{}//data//plot.png'.format(Nexp))
pl.show()
