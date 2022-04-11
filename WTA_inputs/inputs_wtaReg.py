#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:40:26 2020

@author: kat.hendrickson
"""

import numpy as np
# 5 weapons, 4 targets
# Np = 10
# Nd = 5
# Problem Parameters

class WTAinputsReg():
    def __init__(self, num_weapons, num_targets, Pk, V):
        A = np.empty([num_weapons, num_targets * num_weapons], dtype=float)
        for i in range(0,num_weapons):
            col_vec = np.zeros((1, num_weapons))
            col_vec[0,i] = 1
            block = np.repeat(col_vec, num_targets)
            A[i,] = block
        self.A = np.array(A)
        self.b = np.ones(num_weapons)
       
        self.V = V
        self.Q = 1. - Pk
        self.L = np.log(self.Q)
        self.Pk = np.copy(Pk)
        #V = V/np.max(V)
        #L_sum = np.sum(abs(L), axis=0)
    
    def gradPrimal(self,optInputs,x,mu,agent):
        #Pkmax = np.max(self.Pk)
        #Pkmin = np.min(self.Pk)
        #Vmax = np.max(self.V)
        #alpha=4*Vmax*(abs(np.log(1-Pkmax))**2)*((1-Pkmin)**5)
        alpha= 0.01
        Nd = optInputs.m 
        N_targets = int(optInputs.n/optInputs.m)
    
        #x_local = np.reshape(np.copy(x), [Nd, -1])
        #x_local = softmax(x_local, 0)
        
        weapon = int(agent/N_targets)
        target = agent % N_targets
        sumterm = 0.
        for k in range(Nd):
            sumterm += self.L[k, target]*x[k*N_targets+target]
    
        gradient= self.V[target]*self.L[weapon, target]*np.exp(sumterm) + mu @ self.A[:,agent] + alpha*x[weapon*N_targets + target]
        #gradient= V[target]*L[weapon, target]*np.exp(sumterm) + V[target]*(L_sum[target] - abs(L[weapon, target]))*x_local[weapon, target] + mu @ A[:,agent]
        return gradient
    
    def gradDual(self,optInputs,x,mu,agent):
        gradient = self.A[agent,:] @ x - self.b[agent] - optInputs.delta*mu
        return gradient

        # def projPrimal(self,x):
        #    if x > 1.:
        #        x = 1.
        #    if x < 0.:
        #        x = 0.
        #    x_hat = np.copy(x)
        #    return x_hat
    def projPrimal(self, x):
        x = np.where(x > 1., 1., x)
        x = np.where(x < 0., 0., x)
        return x

    # def projDual(self,mu):
    #    if mu < 0:
    #        mu=0
    #    mu_hat=mu
    #    return mu_hat
    def projDual(self, mu):
        mu = np.where(mu < 0, 0, mu)
        return mu
    
            
