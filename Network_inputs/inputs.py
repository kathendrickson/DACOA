#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:40:26 2020

@author: kat.hendrickson
"""

import numpy as np

class NetworkInputs:
# Problem Parameters
    def __init__(self, beta):
        A = np.zeros((66,15))
        A[[0,5,14,1],[0,0,0,0]] = 1
        A[[0,6,3,2,16,15,1],[1,1,1,1,1,1,1]] = 1
        A[[0,7,9,13,2],[2,2,2,2,2]] = 1
        A[[0,4,12,15,1],[3,3,3,3,3]] = 1
        A[[0,6,10,8,11,13,1],[4,4,4,4,4,4,4]] = 1
        A[[17,35,28,27,36,38,18],[5,5,5,5,5,5,5]] = 1
        A[[17,34,33,39,19,20,26,25,18],[6,6,6,6,6,6,6,6,6]] = 1
        A[[17,35,39,37,26,25,18],[7,7,7,7,7,7,7]] = 1
        A[[17,23,22,33,32,31,30,21,20,36,38,18],[8,8,8,8,8,8,8,8,8,8,8,8]] = 1
        A[[17,35,39,29,27,36,24,25,18],[9,9,9,9,9,9,9,9,9]] = 1
        A[[40,50,51,52],[10,10,10,10]] = 1
        A[[40,63,49,52,55,57,41],[11,11,11,11,11,11,11]] = 1
        A[[40,45,46,62,59,56,43,44,53,41],[12,12,12,12,12,12,12,12,12,12]] = 1
        A[[40,63,65,47,48,56,54,64,41],[13,13,13,13,13,13,13,13,13]] = 1
        A[[40,63,61,60,58,56,54,64,41],[14,14,14,14,14,14,14,14,14]] = 1
        self.A = A
        
        self.barray= np.array([50, 50, 23, 32, 15, 39, 31, 11, 38, 16, 35, 11, 20, 36, 26, 23, 27, 50, 50, 33, 10, 19, 19, 37, 31, 27, 15, 18, 7, 39, 33, 25, 30, 12, 26, 10, 5, 11, 28, 13, 50, 50, 50, 39, 37, 32, 5, 32, 22, 18, 30, 15, 36, 38, 8, 20, 35, 34, 8, 16, 11, 12, 39, 35, 6, 39])
        
        self.betaFactor = beta*(11**2)

        self.maxHessianEntry = self.betaFactor
        self.M = np.linalg.norm(A)
        Mj = []
        for j in range(66):
            Mj = np.append(Mj, np.linalg.norm(A[j]))
        self.Mj = Mj
        self.Dx = 10*np.sqrt(15)
        self.B = (-beta * 15 *np.log(11))/(np.min(self.barray))
                
    
    def gradPrimal(self,optInputs,x,mu,agent):
        a=optInputs.xBlocks[agent]  #lower boundary of block (included)
        b=optInputs.xBlocks[agent+1]#upper boundary of block (not included)
        gradient=[]
        for i in range(a,b):
            xi=x[i]
            gradient= np.append(gradient, -self.betaFactor/(1+xi) + mu @ self.A[:,i])
        return gradient
    
    def gradDual(self,optInputs,x,mu,agent):
        a=optInputs.muBlocks[agent]  #lower boundary of block (included)
        b=optInputs.muBlocks[agent+1]#upper boundary of block (not included)
        gradient = self.A[a:b,:] @ x - self.barray[a:b] - optInputs.delta*mu
        return gradient
    
    def projPrimal(self,x):
        for i in range(np.size(x)):
            if x[i] > 10:
                x[i] = 10
            if x[i] < 0:
                x[i] = 0
        x_hat = x
        return x_hat
    
    def projDual(self,mu):
        mu[mu<0]=0
        mu_hat=mu
        return mu_hat

        