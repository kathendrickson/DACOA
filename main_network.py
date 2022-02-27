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

#-----------------------------------------------------------------------------
#       Plot Formatting
#-----------------------------------------------------------------------------

#import matplotlib #uncomment to reset plot styles
#matplotlib.rc_file_defaults() #uncomment to reset plot styles
plt.set_loglevel("error")
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

#-----------------------------------------------------------------------------
#       Use CVXPY to Find "Actual" Solution
#-----------------------------------------------------------------------------
# This is not necessary to run DACOA but is used to plot distance from the
# solution, if desired.

import Network_inputs.cvxpySol_network #inputs file for CVXPY
[xActual, muActual] = Network_inputs.cvxpySol_network.findActual()


#-----------------------------------------------------------------------------
#       Algorithm Parameters
#-----------------------------------------------------------------------------

n = 15         # dimension of primal vector, x
m = 66          # dimension of dual vector, mu



## Use problem parameters to define stepsizes and regularization parameter.        
gamma = .01    # Primal stepsize, bounded by equation (4) in [1] 
delta = .1     # Dual regularization parameter, chosen to reduce reg. error 
rho = delta/(delta ** 2 + 1) # Dual stepsize, bounded in [1].

print("Primal stepsize, gamma:", gamma) 
print("Dual regularization parameter, delta:", delta) 
print("Dual stepsize, rho:", rho) 


#-----------------------------------------------------------------------------
#       Scalars vs Block Runs
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks...")

# Create Inputs Class With Beta = .1 (See [1] or readme for more details.) 
inputs = NetworkInputs(.1)

# Create Np by Nd matrix where each entry i,j is 1 if dual agent j needs 
# updates from prial agent i and is 0 otherwise. We can use the input A matrix 
# to do so for scalar blocks.
scalarDualNeighbors = np.transpose(inputs.A) 

# Create Communication Class with Comm Rate of 0.50 (agents communicate ~50% of the time)
comm50scalar = commClass(.5, scalarDualNeighbors)

# Create DACOA class with inputs defined above.
scalarBlocks = DACOA(delta, gamma, rho, n, m, inputs, comm50scalar)

# Optional: Set the "actual" primal and dual values to compute error later
#           If not set, error will not be calculated.
scalarBlocks.setActual(xActual,muActual)

# Optional: Set the initial primal and dual values
#           If not set, zero vectors will be used.
scalarBlocks.setInit(0*np.ones(n), np.zeros(m))

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks.stopIf(10 ** -6,10**5)

# Run DACOA for scalar blocks
scalarBlocks.run()

#----------------------------------------
## Separable Blocks
print("Running with Separable Blocks...")

# Create non-scalar block arrays : xBlocks is an array containing the first index for each primal block. 
        #Similarly, muBlocks is an array containing the beginning indices for all dual agents.
xBlocks = np.array([0,5,10])    #Primal agent 1 is from 0 to 4, agent 2 is from 5 to 9, and agent 3 is from 10 to the end.
muBlocks = np.array([0,17,40])

# Create dualNeighbors matrix for new block setup. (See above for more discussion.)
blockDualNeighbors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create comm class
comm50blocks = commClass(.5, blockDualNeighbors)

# Create DACOA class with inputs defined above.
vecBlocks = DACOA(delta, gamma, rho, n, m, inputs, comm50blocks)

# Set optional inputs
vecBlocks.setActual(xActual,muActual)
vecBlocks.setInit(0*np.ones(n), np.zeros(m))
vecBlocks.stopIf(10 ** -6,10**5)
vecBlocks.setBlocks(xBlocks,muBlocks)

# Run DACOA for non-scalar blocks
vecBlocks.run()

#----------------------------------------
## Figure Plotting
plt.semilogy(np.arange(1,scalarBlocks.numIter+1), scalarBlocks.iterNorm[1:], color= dark_blue, label="Scalar Blocks")
plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.iterNorm[1:], color= red, linestyle= "dotted", label="Non-Scalar Blocks")
plt.ylabel("$|| x(k) - x(k-1)||$")
plt.xlabel("Time, k")
plt.title("Convergence for Scalar and Non-Scalar Blocks")
plt.legend()
plt.savefig('blocks.eps')
plt.show()

plt.semilogy(np.arange(1,scalarBlocks.numIter+1), scalarBlocks.xError[1:], color= dark_blue, label="Scalar Blocks")
plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.xError[1:], color= red, linestyle= "dotted", label="Non-Scalar Blocks")
plt.ylabel("Distance from CVXPY Solution")
plt.xlabel("Time, k")
plt.title("Error for Scalar and Non-Scalar Blocks")
plt.legend()
#plt.savefig('figure.eps',bbox_inches = "tight",dpi=300)
plt.show()


#-----------------------------------------------------------------------------
#       Varying Communication Rate
#-----------------------------------------------------------------------------

# Using the non-scalar block setup from above:
xBlocks = np.array([0,5,10])
muBlocks = np.array([0,17,40])
blockDualNeighbors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create communication classes for different comm rates:
comm25 = commClass(.25, blockDualNeighbors)
comm50 = commClass(.50, blockDualNeighbors)
comm75 = commClass(.50, blockDualNeighbors)
comm100 = commClass(1, blockDualNeighbors)

# Run for 25% Comms
opt25 = DACOA(delta, gamma, rho, n, m, inputs, comm25)
opt25.setInit(0*np.ones(n), np.zeros(m))
opt25.setActual(xActual,muActual)
opt25.stopIf(10 ** -6,10**5)
opt25.setBlocks(xBlocks,muBlocks)
opt25.run()


# Run for 50% Comms
opt50 = DACOA(delta, gamma, rho, n, m, inputs, comm50)
opt50.setInit(0*np.ones(n), np.zeros(m))
opt50.setActual(xActual,muActual)
opt50.stopIf(10 ** -6,10**5)
opt50.setBlocks(xBlocks,muBlocks)
opt50.run()


# Run for 75% Comms
opt75 = DACOA(delta, gamma, rho, n, m, inputs, comm75)
opt75.setInit(0*np.ones(n), np.zeros(m))
opt75.setActual(xActual,muActual)
opt75.stopIf(10 ** -6,10**5)
opt75.setBlocks(xBlocks,muBlocks)
opt75.run()


# Run for 100% Comms
opt100 = DACOA(delta, gamma, rho, n, m, inputs, comm100)
opt100.setInit(0*np.ones(n), np.zeros(m))
opt100.setActual(xActual,muActual)
opt100.stopIf(10 ** -6,10**5)
opt100.setBlocks(xBlocks,muBlocks)
opt100.run()


#Plotting:

plt.semilogy(np.arange(1,opt25.numIter+1), opt25.iterNorm[1:], color= dark_blue, label="25% Comm. Rate")
plt.semilogy(np.arange(1,opt50.numIter+1), opt50.iterNorm[1:], color= red, linestyle = "dotted", label="50% Comm. Rate")
plt.semilogy(np.arange(1,opt75.numIter+1), opt75.iterNorm[1:], color= dark_green, linestyle = "dashed", label="75% Comm. Rate")
plt.semilogy(np.arange(1,opt100.numIter+1), opt100.iterNorm[1:], color= dark_orange, linestyle = "dashdot", label="100% Comm. Rate")
plt.ylabel("$|| x(k) - x(k-1)||$")
plt.xlabel("Time, k")
plt.title("Communication Rate and Convergence")
plt.legend()
plt.savefig('comm.eps')
plt.show()

plt.semilogy(np.arange(1,opt25.numIter+1), opt25.xError[1:], color= dark_blue, label="25% Comm. Rate")
plt.semilogy(np.arange(1,opt50.numIter+1), opt50.xError[1:], color= red, linestyle = "dotted", label="50% Comm. Rate")
plt.semilogy(np.arange(1,opt75.numIter+1), opt75.xError[1:], color= dark_green, linestyle = "dashed", label="75% Comm. Rate")
plt.semilogy(np.arange(1,opt100.numIter+1), opt100.xError[1:], color= dark_orange, linestyle = "dashdot", label="100% Comm. Rate")
plt.ylabel("Distance from CVXPY Solution")
plt.xlabel("Time, k")
plt.title("Communication Rate and Error")
plt.legend()
#plt.savefig('figure.eps',bbox_inches = "tight",dpi=300)
plt.show()

#-----------------------------------------------------------------------------
#       Varying the Amount of Diagonal Dominance
#-----------------------------------------------------------------------------

inputsBeta25 = NetworkInputs(.25)
inputsBeta50 = NetworkInputs(.50)

beta10 = DACOA(delta, gamma, rho, n, m, inputsBeta25, comm75)
beta10.setInit(0*np.ones(n), np.zeros(m))
beta10.stopIf(10 ** -6,10**5)
beta10.setBlocks(xBlocks,muBlocks)
beta10.run()

beta50 = DACOA(delta, gamma, rho, n, m, inputsBeta50, comm75)
beta50.setInit(0*np.ones(n), np.zeros(m))
beta50.stopIf(10 ** -6,10**5)
beta50.setBlocks(xBlocks,muBlocks)
beta50.run()


plt.semilogy(np.arange(1,vecBlocks.numIter+1), vecBlocks.iterNorm[1:], color= dark_blue, label="Beta = .10")
plt.semilogy(np.arange(1,beta10.numIter+1), beta10.iterNorm[1:], color= red, linestyle= "dotted",  label="Beta = .25")
plt.semilogy(np.arange(1,beta50.numIter+1), beta50.iterNorm[1:], color= dark_green, linestyle= "dashed", label="Beta = .50")
plt.ylabel("$|| x(k) - x(k-1)||$")
plt.xlabel("Time, k")
plt.title("Diagonal Dominance and Convergence")
plt.legend()
plt.savefig('beta.eps')
plt.show()