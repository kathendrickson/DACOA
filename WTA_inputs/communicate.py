#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:09:38 2020

@author: kat.hendrickson
"""

import numpy as np

class commClass():
    def __init__(self, commRate):
        self.commRate = commRate

    def comm(self, optInputs, Xp, Xd):
        Np = optInputs.Np
        Nd = optInputs.Nd
        
        B=np.random.rand(Np, Np+Nd)
        X_new = np.concatenate((Xp,Xd),axis=1)
        dup = np.zeros((Np,Nd))
        
        for i in range(Np):
            a=optInputs.xBlocks[i]  #lower boundary of block (included)
            b=optInputs.xBlocks[i+1]#upper boundary of block (not included)
            for j in range(Np+Nd):
                if i != j:
                    if B[i,j] <= self.commRate:
                        B[i,j] = 1
                        X_new[a:b,j] = np.copy(Xp[a:b,i])
                        if j >= Np:
                            dup[i,j-Np] = 1
        
        Xp_new = X_new[:,0:Np]
        Xd_new = X_new[:,Np:Np+Nd]
        
        return Xp_new, Xd_new, dup
