#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:40:26 2020

@author: kat.hendrickson
"""

import numpy as np


# Problem Parameters 
A = np.array([[-1, 0, -3, 0, 0, 4, 0, 0, 10, 0], 
              [0, 1, 5, 1, 1, 0, 0, 2, 0, 5],
              [0, 0, 1, 1, -5, 1, 4, 0, 0, 0],
              [0, 0, -2, 0, 0, 8, 1, 1, -3, 1],
              [0, 0, 0, 0, -3, 0, 1, 1, 1, 0],
              [0, 4, 0, 0, 0, 0, 0, 2, 1, -4]])
barray= np.array([-2,4,-10,5,1,8])

def gradPrimal(self,x,mu,agent):
    a=self.xBlocks[agent]  #lower boundary of block (included)
    b=self.xBlocks[agent+1]#upper boundary of block (not included)
    gradient=[]
    for i in range(a,b):
        xi=x[i]
        sumterm=0
        for j in range(self.n):     #figure this out
            if j != (i) :
                sumterm=sumterm + 4*(xi - x[j])
        gradient= np.append(gradient, 4*(xi**3) + (1/20)*sumterm + mu @ A[:,i])
    return gradient

def gradDual(self,x,mu,agent):
    a=self.muBlocks[agent]  #lower boundary of block (included)
    b=self.muBlocks[agent+1]#upper boundary of block (not included)
    gradient = A[a:b,:] @ x - barray[a:b] - self.delta*mu
    return gradient

def projPrimal(x):
    for i in range(np.size(x)):
        if x[i] > 10:
            x[i] = 10
        if x[i] < 0.1:
            x[i] = 0.1
    x_hat = x
    return x_hat

def projDual(mu):
    mu[mu<0]=0
    mu_hat=mu
    return mu_hat

        