#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from algorithm import DACOA
from Network_inputs.inputs import NetworkInputs
from Network_inputs.communicate import commClass

#the following code resets the plot styles
#import matplotlib
#matplotlib.rc_file_defaults()

plt.rcParams["font.family"] = "Times New Roman"
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":18,
                                "axes.labelsize":16,
                                "figure.figsize": 6.4,
                                "savefig.dpi":600,     
                                "savefig.format": "eps",
                                "savefig.bbox": "tight",
                                "savefig.pad_inches": 0.1
                                })
sns.despine()
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
pink = "#fb9a99"
red = "#e31a1c"
light_orange = "#fdbf6f"
dark_orange = "#ff7f00"
light_purple = "#cab2d6"
dark_purple = "#6a3d9a"

# Algorithm Parameters
n = 15         # number of primal agents
m = 66          # number of dual agents
delta = .1   # dual regularization parameter
rho = delta/(delta ** 2 + 2) # dual step-size
gamma = .01    # primal step-size

# Find "Actual" Solution
import Network_inputs.cvxpySol_network
[xActual, muActual] = Network_inputs.cvxpySol_network.findActual()


#-----------------------------------------------------------------------------
#       Scalars vs Block Runs
#-----------------------------------------------------------------------------


inputs = NetworkInputs(1)
comm75 = commClass(.75)


print("Running with Scalar Blocks...")
scalarBlocks = DACOA(delta, gamma, rho, n, m, inputs, comm75)
scalarBlocks.setActual(xActual,muActual)
scalarBlocks.setInit(0*np.ones(n), np.zeros(m))
scalarBlocks.stopIf(10 ** -6,10**5,flagIter=1)
scalarBlocks.run()

print("Running with Separable Blocks...")
xBlocks = np.array([0,5,10])
muBlocks = np.array([0,17,40])
vecBlocks = DACOA(delta, gamma, rho, n, m, inputs, comm75)
vecBlocks.setActual(xActual,muActual)
vecBlocks.setInit(0*np.ones(n), np.zeros(m))
vecBlocks.stopIf(10 ** -6,10**5,flagIter=1)
vecBlocks.defBlocks(xBlocks,muBlocks)
vecBlocks.run()

plt.semilogy(np.arange(1,scalarBlocks.numIter+1), scalarBlocks.iterNorm[1:], color= dark_blue, label="Scalar Blocks")
plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.iterNorm[1:], color= red, linestyle= "dotted", label="Non-Scalar Blocks")
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Convergence for Scalar and Non-Scalar Blocks")
plt.legend()
plt.savefig('blocks.eps')
plt.show()

# plt.semilogy(np.arange(1,scalarBlocks.numIter+1), scalarBlocks.xError[1:], color= dark_blue, label="Scalar Updates")
# plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.xError[1:], color= red, linestyle= "dotted", label="Block Updates")
# plt.ylabel("Distance from CVXPY Solution")
# plt.xlabel("Iteration Number")
# plt.title("Error as a Function of Iteration Number")
# plt.legend()
# #plt.savefig('figure.eps',bbox_inches = "tight",dpi=300)
# plt.show()

#-----------------------------------------------------------------------------
#       Varying Communication Rate
#-----------------------------------------------------------------------------

#Running:
iterNums = np.array([])
xBlocks = np.array([0,5,10])
muBlocks = np.array([0,17,40])

#25% Comm Rate
comm25 = commClass(.25)
comm50 = commClass(.50)
comm100 = commClass(1)

opt25 = DACOA(delta, gamma, rho, n, m, inputs, comm25)
opt25.setInit(0*np.ones(n), np.zeros(m))
opt25.setActual(xActual,muActual)
opt25.stopIf(10 ** -6,10**5,flagIter=1)
opt25.defBlocks(xBlocks,muBlocks)
opt25.run()
iterNorm25 = opt25.iterNorm
numIter25 = opt25.numIter
xError25 = opt25.xError

opt50 = DACOA(delta, gamma, rho, n, m, inputs, comm50)
opt50.setInit(0*np.ones(n), np.zeros(m))
opt50.setActual(xActual,muActual)
opt50.stopIf(10 ** -6,10**5,flagIter=1)
opt50.defBlocks(xBlocks,muBlocks)
opt50.run()
iterNorm50 = opt50.iterNorm
numIter50 = opt50.numIter
xError50 = opt50.xError

opt75 = DACOA(delta, gamma, rho, n, m, inputs, comm75)
opt75.setInit(0*np.ones(n), np.zeros(m))
opt75.setActual(xActual,muActual)
opt75.stopIf(10 ** -6,10**5,flagIter=1)
opt75.defBlocks(xBlocks,muBlocks)
opt75.run()
iterNorm75 = opt75.iterNorm
numIter75 = opt75.numIter
xError75 = opt75.xError

opt100 = DACOA(delta, gamma, rho, n, m, inputs, comm100)
opt100.setInit(0*np.ones(n), np.zeros(m))
opt100.setActual(xActual,muActual)
opt100.stopIf(10 ** -6,10**5,flagIter=1)
opt100.defBlocks(xBlocks,muBlocks)
opt100.run()
iterNorm100 = opt100.iterNorm
numIter100 = opt100.numIter
xError100 = opt100.xError

#Plotting:

plt.semilogy(np.arange(1,numIter25+1), iterNorm25[1:], color= dark_blue, label="25% Comm. Rate")
plt.semilogy(np.arange(1,numIter50+1), iterNorm50[1:], color= red, linestyle = "dotted", label="50% Comm. Rate")
plt.semilogy(np.arange(1,numIter75+1), iterNorm75[1:], color= dark_green, linestyle = "dashed", label="75% Comm. Rate")
plt.semilogy(np.arange(1,numIter100+1), iterNorm100[1:], color= dark_orange, linestyle = "dashdot", label="100% Comm. Rate")
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Communication Rate and Convergence")
plt.legend()
plt.savefig('comm.eps')
plt.show()

# plt.semilogy(np.arange(1,numIter25+1), xError25[1:], color= dark_blue, label="25% Comm. Rate")
# plt.semilogy(np.arange(1,numIter50+1), xError50[1:], color= red, linestyle = "dotted", label="50% Comm. Rate")
# plt.semilogy(np.arange(1,numIter75+1), xError75[1:], color= dark_green, linestyle = "dashed", label="75% Comm. Rate")
# plt.semilogy(np.arange(1,numIter100+1), xError100[1:], color= dark_orange, linestyle = "dashdot", label="100% Comm. Rate")
# plt.ylabel("Distance from CVXPY Solution")
# plt.xlabel("Iteration Number")
# plt.title("Error as a Function of Iteration Number")
# plt.legend()
# #plt.savefig('figure.eps',bbox_inches = "tight",dpi=300)
# plt.show()

#-----------------------------------------------------------------------------
#       Varying Beta
#-----------------------------------------------------------------------------

inputsBeta10 = NetworkInputs(10)
inputsBeta50 = NetworkInputs(50)

beta10 = DACOA(delta, gamma, rho, n, m, inputsBeta10, comm75)
beta10.setActual(xActual,muActual)
beta10.setInit(0*np.ones(n), np.zeros(m))
beta10.stopIf(10 ** -6,10**5,flagIter=1)
beta10.defBlocks(xBlocks,muBlocks)
beta10.run()

beta50 = DACOA(delta, gamma, rho, n, m, inputsBeta50, comm75)
beta50.setActual(xActual,muActual)
beta50.setInit(0*np.ones(n), np.zeros(m))
beta50.stopIf(10 ** -6,10**5,flagIter=1)
beta50.defBlocks(xBlocks,muBlocks)
beta50.run()


plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.iterNorm[1:], color= dark_blue, label="Beta = 1")
plt.semilogy(np.arange(1,beta10.numIter+1), beta10.iterNorm[1:], color= red, linestyle= "dotted",  label="Beta = 10")
plt.semilogy(np.arange(1,beta50.numIter+1), beta50.iterNorm[1:], color= dark_green, linestyle= "dashed", label="Beta = 50")
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Diagonal Dominance and Convergence")
plt.legend()
plt.savefig('beta.eps')
plt.show()