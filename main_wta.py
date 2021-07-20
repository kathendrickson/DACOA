#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

import numpy as np
<<<<<<< Updated upstream

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
=======
import numpy.linalg as la
from algorithm import DACOA

# Import inputs file.
from WTA_inputs.inputs_wta import WTAinputs
from WTA_inputs.inputs_wtaReg import WTAinputsReg
from WTA_inputs.communicate import commClass

# Problem Parameters
num_weapons = 10
num_targets = 7
my_number = 4
n = num_weapons * num_targets    #size of primal variable
m = num_weapons                       #size of dual variable
pBlocks = num_targets*np.arange(num_weapons)

##############################################################################
## Run DACOA and Compare to Brute Force

#Algorithm Parameters
delta = .1 #dual regularization parameter
rho = delta/(delta ** 2 + 2)    #dual stepsize
gamma = .1 #primal stepsize

normDiff = [0]
evDiff = [0]

# Pk =np.array([[0.5, 0.3, 0.4, 0.3],
#                 #[.2, .6, .3, .4],
#                 [.6, .3, .2, .1],
#                 [.3, .2, .5, .7],
#                 [.2, .4, .7, .6]])
# V = np.array([5, 8, 4, 5])

Pk= np.array([[ 0.64463895,  0.59598386,  0.62282959,  0.69496994,  0.7438825,   0.65428474,  0.61025209],
  [ 0.58094327,  0.74144878,  0.61107755,  0.6507706,   0.53359042,  0.52685467,  0.51058193],
  [ 0.5072811,   0.72318633,  0.56037266,  0.53998156,  0.7318601,   0.54530808,   0.5300151 ],
  [ 0.72076566,  0.55156122,  0.71958498,  0.71594228,  0.59875791,  0.61800393,  0.50001622],
  [ 0.56475473,  0.69762735,  0.51372416,  0.50514041,  0.58957253,  0.6199411,    0.61018262],
  [ 0.56432397,  0.54127577,  0.53176584,  0.72068753,  0.56219193,  0.6830076,    0.7047951 ],
  [ 0.54787719,  0.69430458,  0.58761586,  0.72638621,  0.57360651,  0.55029484,   0.74306214],
  [ 0.55249685,  0.62723239,  0.55114336,  0.6895358,   0.69889258,  0.54854262,    0.62703139],
  [ 0.68568259,  0.68873851,  0.65304307 , 0.70979717 , 0.71746624,  0.57047922,    0.51426986],
  [ 0.66770988,  0.51893396,  0.61396158,  0.61292287,  0.6911437,   0.65894004,   0.51537253]])
V=np.array( [ 9.05605879,  8.07168717,  5.39014167,  6.42463123,  6.85307702,  6.51115109, 7.807034 ])

for k in range(1,2):
    print("~~~")
    print(k)
    #Pk = np.random.random((5, 4))*0.25 + 0.5
    #V = np.reshape(np.random.random((1, 4))*5. + 5, (-1))
    inputs=WTAinputs(num_weapons, num_targets, Pk,V)
    comm =commClass(1)
    
    #DACOA Setup
    opt = DACOA(delta,gamma,rho, n, m, inputs, comm)
    #opt.inputFiles("inputs_wta","WTA_inputs.communicate")
    opt.defBlocks(pBlocks,np.arange(m))
    opt.useScalars()
    opt.setInit(.5*np.ones(n), np.zeros(m))
    opt.stopIf(10e-6,25000,1)
    
    #DACOA Run and Post-Processing
    opt.run()
    xUpdated = np.copy(opt.xFinal)
    xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
    #print(xUpdated)
    selection = np.argmax(xUpdated, axis=1)
    print("DACOA:")
    print(selection)
    
    xOrig = np.copy(xUpdated)
    
    
    # #Brute Force Computation and Procession
    # achieved_comp = np.ones((num_targets,1));
    # for i in range(num_weapons):
    #   achieved_comp[selection[i]] = achieved_comp[selection[i]]*inputs.Q[i, selection[i]]
    # valueOrig = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    # print(valueOrig)
    # print("~~~")
    # print("Opt:")
    # EV = 100.
    # index = 0.
    # for i in range(int(pow(num_targets, num_weapons))):
    #   achieved_comp = np.ones((num_targets,1));
    #   remainder = i;
    #   for j in range(num_weapons):
    #     target = int(np.floor(remainder/pow(num_targets, num_weapons-j-1)));
    
    #     remainder = np.mod(remainder, pow(num_targets, num_weapons-j-1));
    #     #print("W: %d \t T: %d"%(j, target))
    #     achieved_comp[target] = achieved_comp[target]*inputs.Q[j, target];
    
    #   value = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    
    #   if value < EV:
    #     EV = value
    #     index = i
    # actual = np.zeros((num_weapons,1)).astype(int)
    # remainder = index
    # for j in range(num_weapons):
    #   actual[j] = int(np.floor(remainder/pow(num_targets, num_weapons-j-1)));
    #   remainder = np.mod(remainder, pow(num_targets, num_weapons-j-1));
    # print(np.reshape(actual,(-1)))
    # print(EV)

    
    ##############################################################################
    ## Run DACOA with Regularization
    
    #Import regularized inputs file
    inputsReg=WTAinputsReg(num_weapons, num_targets, Pk, V)
    
    
    #DACOA Setup
    opt = DACOA(delta,gamma,rho, n, m, inputsReg, comm)
    #opt.inputFiles("inputs_wtaReg","WTA_inputs.communicate")
    opt.defBlocks(pBlocks,np.arange(m))
    opt.useScalars()
    opt.setInit(.5*np.ones(n), np.zeros(m))
    opt.stopIf(10e-6,25000,1)
    
    #DACOA Run and Post-Processing
    opt.run()
    xUpdated = np.copy(opt.xFinal)
    xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
    #print(xUpdated)
    selection = np.argmax(xUpdated, axis=1)
    print("~~~")
    print("DACOA with Reg:")
    print(selection)
   
    
    xReg = np.copy(xUpdated)
    achieved_comp = np.ones((num_targets,1));
    for i in range(num_weapons):
      achieved_comp[selection[i]] = achieved_comp[selection[i]]*inputs.Q[i, selection[i]]
    valueReg = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    print(valueReg)
    print(opt.numIter)
    normDiff.append(la.norm(xOrig - xReg))
    #evDiff.append(np.array([valueOrig-EV, valueReg - EV, (valueReg-EV)/EV]))
>>>>>>> Stashed changes
