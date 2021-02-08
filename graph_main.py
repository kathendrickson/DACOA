#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

import numpy as np

# Algorithm Parameters
n = 15         # number of primal agents
m = 66          # number of dual agents
delta = .1   # dual regularization parameter
rho = delta/(delta ** 2 + 2) # dual step-size
gamma = .01    # primal step-size

# Find "Actual" Solution
#import graph_cvxpySol
#[xActual, muActual] = graph_cvxpySol.findActual()

from algorithm import DACOA

print("Running with Scalar Blocks...")
xScalar = DACOA(delta, gamma, rho, n, m)

xScalar.inputFiles("graph_inputs","communicate")

#xScalar.setActual(xActual,muActual)

xScalar.setInit(0*np.ones(n), np.zeros(m))

xScalar.stopIf(10 ** -8,10**5,flagIter=1)

xScalar.run()

print("Number of iterations: ", xScalar.numIter)
print("Final primal variable: ", xScalar.xFinal)
print("Final dual variable: ", xScalar.muFinal)

# print("Running with Clouds (blocks of size n and m)...")
# xClouds = DACOA(delta, gamma, rho, n, m)

# xClouds.setActual(xActual,muActual)

# xClouds.setInit(10*np.ones(n), np.zeros(m))

# xClouds.defBlocks([0],[0])

# xClouds.stopIf(10 ** -8,10**4,flagIter=1)

# xClouds.run()

# print("Number of iterations: ", xClouds.numIter)
# print("Final primal variable: ", xClouds.xFinal)

# print("Running with Single Gradient Function Option...")
# xSG = DACOA(delta, gamma, rho, n, m)

# xSG.setActual(xActual,muActual)

# xSG.setInit(10*np.ones(n), np.zeros(m))

# xSG.defBlocks([0],[0])

# xSG.stopIf(10 ** -8,10**4,flagIter=1)

# xSG.useScalars()

# xSG.inputFiles("inputs_singlegrad","communicate")

# xSG.run()

# print("Number of iterations: ", xSG.numIter)
# print("Final primal variable: ", xSG.xFinal)

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

