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

gamma = .1    # dual step-size


from algorithm import DACOA

for delta in [.0001, .001, .01, .1, 1, 10]:
    rho = delta/(delta ** 2 + 2) # primal step-size
    
    for gamma in [.0001, .001, .01, .1, 1, 10]:
    
        xScalar = DACOA(delta, gamma, rho, n, m)

        xScalar.inputFiles("inputs_wta","communicate")

        xScalar.setInit(.5*np.ones(n),np.zeros(m))

        xScalar.stopIf(10 ** -8,10**4,flagIter=1)

        xScalar.useScalars()

        xScalar.run()
    
        print("delta = ", delta, "gamma = ", gamma)
        print("Number of iterations: ", xScalar.numIter)
        print("Final primal variable: ", xScalar.xFinal)

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

