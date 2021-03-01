#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import graph_cvxpySol
[xActual, muActual] = graph_cvxpySol.findActual()

from algorithm import DACOA




#-----------------------------------------------------------------------------
#       Scalars vs Block Runs
#-----------------------------------------------------------------------------

print("Running with Scalar Blocks...")
xScalar = DACOA(delta, gamma, rho, n, m)
xScalar.inputFiles("graph_inputs","communicate")
xScalar.setActual(xActual,muActual)
xScalar.setInit(0*np.ones(n), np.zeros(m))
xScalar.stopIf(10 ** -6,10**5,flagIter=1)
xScalar.setCommRate(0.75)
xScalar.run()

print("Running with Separable Blocks...")
xBlocks = np.array([0,5,10])
muBlocks = np.array([0,17,40])
xSep = DACOA(delta, gamma, rho, n, m)
xSep.inputFiles("graph_inputs","communicate")
xSep.setActual(xActual,muActual)
xSep.setInit(0*np.ones(n), np.zeros(m))
xSep.stopIf(10 ** -6,10**5,flagIter=1)
xSep.defBlocks(xBlocks,muBlocks)
xSep.setCommRate(0.75)
xSep.run()

plt.semilogy(np.arange(1,xScalar.numIter+1), xScalar.iterNorm[1:], color= dark_blue, label="Scalar Blocks")
plt.semilogy(np.arange(1,xSep.numIter+1), xSep.iterNorm[1:], color= red, linestyle= "dotted", label="Non-Scalar Blocks")
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Convergence for Scalar and Non-Scalar Blocks")
plt.legend()
plt.savefig('blocks.eps')
plt.show()

# plt.semilogy(np.arange(1,xScalar.numIter+1), xScalar.xError[1:], color= dark_blue, label="Scalar Updates")
# plt.semilogy(np.arange(1,xSep.numIter+1), xSep.xError[1:], color= red, linestyle= "dotted", label="Block Updates")
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

for CR in [.25, .5, .75, 1]:
    xBlocks = np.array([0,5,10])
    muBlocks = np.array([0,17,40])
    xComms = DACOA(delta, gamma, rho, n, m)
    xComms.inputFiles("graph_inputs","communicate")
    xComms.setInit(0*np.ones(n), np.zeros(m))
    xComms.setActual(xActual,muActual)
    xComms.stopIf(10 ** -6,10**5,flagIter=1)
    xComms.defBlocks(xBlocks,muBlocks)
    xComms.setCommRate(CR)
    xComms.run()
    iterNums = np.append(iterNums,xComms.numIter)
    if CR == 0.25:
        iterNorm25 = xComms.iterNorm
        numIter25 = xComms.numIter
        xError25 = xComms.xError
    elif CR == 0.5:
        iterNorm50 = xComms.iterNorm
        numIter50 = xComms.numIter
        xError50 = xComms.xError
    elif CR == 0.75:
        iterNorm75 = xComms.iterNorm
        numIter75 = xComms.numIter
        xError75 = xComms.xError
    elif CR == 1.0:
        iterNorm100 = xComms.iterNorm
        numIter100 = xComms.numIter
        xError100 = xComms.xError

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


x10 = DACOA(delta, gamma, rho, n, m)
x10.inputFiles("graph10_inputs","communicate")
x10.setActual(xActual,muActual)
x10.setInit(0*np.ones(n), np.zeros(m))
x10.stopIf(10 ** -6,10**5,flagIter=1)
x10.defBlocks(xBlocks,muBlocks)
x10.run()

x50 = DACOA(delta, gamma, rho, n, m)
x50.inputFiles("graph50_inputs","communicate")
x50.setActual(xActual,muActual)
x50.setInit(0*np.ones(n), np.zeros(m))
x50.stopIf(10 ** -6,10**5,flagIter=1)
x50.defBlocks(xBlocks,muBlocks)
x50.run()


plt.semilogy(np.arange(1,xSep.numIter+1), xSep.iterNorm[1:], color= dark_blue, label="Beta = 1")
plt.semilogy(np.arange(1,x10.numIter+1), x10.iterNorm[1:], color= red, linestyle= "dotted",  label="Beta = 10")
plt.semilogy(np.arange(1,x50.numIter+1), x50.iterNorm[1:], color= dark_green, linestyle= "dashed", label="Beta = 50")
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Diagonal Dominance and Convergence")
plt.legend()
plt.savefig('beta.eps')
plt.show()

#-----------------------------------------------------------------------------
#       Averaging Smart and Dumb Blocks
#-----------------------------------------------------------------------------
# smartIterAvgs=[]
# ranIterAvgs=[]
# for CR in np.arange(.1,1.1,.2):
#     smartIterNums = []
#     ranIterNums = []
#     for i in np.arange(1,11):
#         xBlocks = np.array([0,5,10])
#         muBlocks = np.array([0,17,40])

#         xSmart = DACOA(delta, gamma, rho, n, m)
#         xSmart.inputFiles("graph_inputs","communicate")
#         xSmart.setInit(0*np.ones(n), np.zeros(m))
#         xSmart.stopIf(10 ** -6,10**5,flagIter=1)
#         xSmart.defBlocks(xBlocks,muBlocks)
#         xSmart.setCommRate(CR)
#         xSmart.run()
#         smartIterNums = np.append(smartIterNums,xSmart.numIter)
        
#         muBlocks = np.array([0,22,44])
#         xRan = DACOA(delta, gamma, rho, n, m)
#         xRan.inputFiles("graph2_inputs","communicate")
#         xRan.setInit(0*np.ones(n), np.zeros(m))
#         xRan.stopIf(10 ** -6,10**5,flagIter=1)
#         xRan.defBlocks(xBlocks,muBlocks)
#         xRan.setCommRate(CR)
#         xRan.run()
#         ranIterNums = np.append(ranIterNums,xRan.numIter)
    
#     smartIterAvgs = np.append(smartIterAvgs,np.mean(smartIterNums))
#     ranIterAvgs = np.append(ranIterAvgs,np.mean(ranIterNums))

#-----------------------------------------------------------------------------
#       Smart vs Dumb Block Errors
#-----------------------------------------------------------------------------


# print("Running with Separable Blocks...")
# xBlocks = np.array([0,5,10])
# muBlocks = np.array([0,17,40])
# xSep = DACOA(delta, gamma, rho, n, m)
# xSep.inputFiles("graph_inputs","communicate")
# xSep.setActual(xActual,muActual)
# xSep.setInit(0*np.ones(n), np.zeros(m))
# xSep.stopIf(10 ** -6,10**5,flagIter=1)
# xSep.defBlocks(xBlocks,muBlocks)
# xSep.setCommRate(0.5)
# xSep.run()

# print("Running with Random Blocks...")
# import graph2_cvxpySol
# [xActual2, muActual2] = graph2_cvxpySol.findActual()
# xBlocks = np.array([0,5,10])
# muBlocks = np.array([0,22,44])
# xRan = DACOA(delta, gamma, rho, n, m)
# xRan.inputFiles("graph2_inputs","communicate")
# xRan.setActual(xActual2,muActual2)
# xRan.setInit(0*np.ones(n), np.zeros(m))
# xRan.stopIf(10 ** -6,10**5,flagIter=1)
# xRan.defBlocks(xBlocks,muBlocks)
# xRan.setCommRate(0.5)
# xRan.run()

# plt.semilogy(np.arange(1,xRan.numIter+1), xRan.iterNorm[1:], color= dark_blue, label="Scalar Updates")
# plt.semilogy(np.arange(1,xSep.numIter+1), xSep.iterNorm[1:], color= red, linestyle= "dotted", label="Block Updates")
# plt.ylabel("Distance Between Iterations")
# plt.xlabel("Iteration Number")
# plt.title("Convergence as a Function of Iteration Number")
# plt.legend()
# #plt.savefig('figure.eps')
# plt.show()

# plt.semilogy(np.arange(1,xRan.numIter+1), xRan.xError[1:], color= dark_blue, label="Scalar Updates")
# plt.semilogy(np.arange(1,xSep.numIter+1), xSep.xError[1:], color= red, linestyle= "dotted", label="Block Updates")
# plt.ylabel("Distance from CVXPY Solution")
# plt.xlabel("Iteration Number")
# plt.title("Error as a Function of Iteration Number")
# plt.legend()
# #plt.savefig('figure.eps',bbox_inches = "tight",dpi=300)
# plt.show()
