#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

import numpy as np

# Algorithm Parameters
n = 10         # number of primal agents
m = 5          # number of dual agents
#delta = 0.001   # dual regularization parameter

gamma = .1    # primal step-size


from algorithm import DACOA

# for delta in [.0001, .001, .01, .1, 1, 10]:
#     rho = delta/(delta ** 2 + 2) # dual step-size
    
#     for gamma in [.0001, .001, .01, .1, 1, 10]:
    
#         xScalar = DACOA(delta, gamma, rho, n, m)

#         xScalar.inputFiles("inputs_wta","communicate")

#         xScalar.setInit(.5*np.ones(n),np.zeros(m))

#         xScalar.stopIf(10 ** -8,10**4,flagIter=1)

#         xScalar.useScalars()

#         xScalar.run()
    
#         print("delta = ", delta, "gamma = ", gamma)
#         print("Number of iterations: ", xScalar.numIter)
#         print("Final primal variable: ", xScalar.xFinal)

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_context("notebook", rc={"font.size":16,
#                                 "axes.titlesize":20,
#                                 "axes.labelsize":16})
# sns.set_palette("pastel")

# sns.despine()

# plt.semilogy(np.arange(1,xScalar.numIter+1), xScalar.iterNorm[1:])
# plt.ylabel("Distance Between Iterations")
# plt.xlabel("Iteration Number")
# plt.title("Convergence as a Function of Iteration Number")
# plt.show()

from inputs_wta import inputs

inputs=inputs()
delta = .01
rho = delta/(delta ** 2 + 2)
gamma = 1

num_weapons = 5
num_targets = 2
my_number = 4
n = num_weapons * num_targets    #size of primal variable
m = num_weapons                       #size of dual variable
pBlocks = num_targets*np.arange(num_weapons)
        
opt = DACOA(delta,gamma,rho, n, m, inputs)

opt.inputFiles("inputs_wta","communicate")

opt.defBlocks(pBlocks,np.arange(m))
opt.useScalars()

a=opt.xBlocks[my_number]  #lower boundary of primal block (included)
b=opt.xBlocks[my_number+1] #upper boundary of primal block (not included)

mu = np.zeros(m)  #shared in primal and dual updates
x = .5*np.ones(n) #shared in primal and dual updates
xUpdated = opt.singlePrimal(x, mu, my_number,inputs)
muUpdated = opt.singleDual(x,mu[my_number],my_number,inputs)
print(xUpdated)