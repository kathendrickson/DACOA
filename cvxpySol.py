#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:16:47 2020

@author: kat.hendrickson
"""
import cvxpy as cp
import numpy as np

def findActual():
    A = np.array([[-1, 0, -3, 0, 0, 4, 0, 0, 10, 0], 
                  [0, 1, 5, 1, 1, 0, 0, 2, 0, 5],
                  [0, 0, 1, 1, -5, 1, 4, 0, 0, 0],
                  [0, 0, -2, 0, 0, 8, 1, 1, -3, 1],
                  [0, 0, 0, 0, -3, 0, 1, 1, 1, 0],
                  [0, 4, 0, 0, 0, 0, 0, 2, 1, -4]])
    b= np.array([-2,4,-10,5,1,8])
    Np=10
    Nd=6
    
    x=cp.Variable(Np)
    mu=cp.Variable(Nd)
    
    
    constraints=[A@x <= b,
                 x>= .1,
                 x<=10]
    
    sum = 0
    sum2 = 0
    for i in range(Np):
        for j in range(Np):
            if j != i:
                sum2=sum2 + (x[i]-x[j])**2
        sum=sum + x[i]**4 + (1/20)*sum2
    
    OrigF=sum 
    #
    
    
    obj = cp.Minimize(OrigF)
    
    prob = cp.Problem(obj,constraints)
    prob.solve()
    
    # print("\nThe optimal value is", prob.value)
    # print("A solution x is")
    # print(x.value)
    # print("A dual solution corresponding to the inequality constraints is")
    # print(prob.constraints[0].dual_value)

    return x.value, prob.constraints[0].dual_value