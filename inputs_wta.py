#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:40:26 2020

@author: kat.hendrickson
"""

import numpy as np
from scipy.special import softmax
# 5 weapons, 2 targets
# Np = 10
# Nd = 5
# Problem Parameters



A = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],])
b= np.array([1, 1, 1, 1, 1])
Pk = np.array([ [0.75, 0.25],
				[0.65, 0.35],
				[0.55, 0.45],
				[0.25, 0.75],
				[0.45, 0.65]])
Q = 1. - Pk
L = np.log(Q)
V = np.array([1., 1.])
#V = V/np.max(V)
#L_sum = np.sum(abs(L), axis=0)

def gradPrimal(self,x,mu,agent):
    Nd = self.m
    N_targets = int(self.n/self.m)

    #x_local = np.reshape(np.copy(x), [Nd, -1])
    #x_local = softmax(x_local, 0)
    
    weapon = int(agent/N_targets)
    target = agent % N_targets
    sumterm = 0.
    for k in range(Nd):
        sumterm += L[k, target]*x[k*N_targets+target]

    gradient= V[target]*L[weapon, target]*np.exp(sumterm) + mu @ A[:,agent]
    #gradient= V[target]*L[weapon, target]*np.exp(sumterm) + V[target]*(L_sum[target] - abs(L[weapon, target]))*x_local[weapon, target] + mu @ A[:,agent]
    return gradient

def gradDual(self,x,mu,agent):
    gradient = A[agent,:] @ x - b[agent] - self.delta*mu
    return gradient

def projPrimal(x):
    if x > 1.:
        x = 1.
    if x < 0.:
        x = 0.
    x_hat = np.copy(x)
    return x_hat

def projDual(mu):
    if mu < 0:
        mu=0
    mu_hat=mu
    return mu_hat

        