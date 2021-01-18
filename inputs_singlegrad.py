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
    xi=x[agent]
    sumterm=0
    for j in range(self.n):
        if j != (agent) :
            sumterm=sumterm + 4*(xi - x[j])
    gradient= 4*(xi**3) + (1/20)*sumterm + mu @ A[:,agent]
    return gradient

def gradDual(self,x,mu,agent):
    gradient = A[agent,:] @ x - barray[agent] - self.delta*mu
    return gradient

def projPrimal(x):
    if x > 10:
        x = 10
    if x < 0.1:
        x = 0.1
    x_hat = x
    return x_hat

def projDual(mu):
    if mu < 0:
        mu=0
    mu_hat=mu
    return mu_hat

        