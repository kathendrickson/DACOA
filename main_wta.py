#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:38:32 2020

@author: kat.hendrickson
"""

#TODO
            #xUpdated = np.reshape(xUpdated, (Nd, int(self.n/Nd)))
            #primals.append(xUpdated)
            #selection.append(np.argmax(xUpdated, axis=1))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("notebook", rc={"font.size":16,
                            "axes.titlesize":20,
                            "axes.labelsize":16})
sns.set_palette("pastel")
sns.despine()

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

#from inputs_wta import inputs

# inputs=inputs()
# delta = .01
# rho = delta/(delta ** 2 + 2)
# gamma = 1

# num_weapons = 5
# num_targets = 2
# my_number = 4
# n = num_weapons * num_targets    #size of primal variable
# m = num_weapons                       #size of dual variable
# pBlocks = num_targets*np.arange(num_weapons)
        
# opt = DACOA(delta,gamma,rho, n, m, inputs)

# opt.inputFiles("inputs_wta","communicate")

# opt.setBlocks(pBlocks,np.arange(m))
# opt.useScalars()

# a=opt.xBlocks[my_number]  #lower boundary of primal block (included)
# b=opt.xBlocks[my_number+1] #upper boundary of primal block (not included)

# mu = np.zeros(m)  #shared in primal and dual updates
# x = .5*np.ones(n) #shared in primal and dual updates
# xUpdated = opt.singlePrimal(x, mu, my_number,inputs)
# muUpdated = opt.singleDual(x,mu[my_number],my_number,inputs)
# print(xUpdated)

import numpy.linalg as la
from algorithm import DACOA

# Import inputs file.
from WTA_inputs.inputs_wta import WTAinputs
from WTA_inputs.inputs_wtaReg import WTAinputsReg
from WTA_inputs.communicate import commClass

# Problem Parameters
num_weapons = 5
num_targets = 4
my_number = 5
n = num_weapons * num_targets    #size of primal variable
m = num_weapons                       #size of dual variable
pBlocks = num_targets*np.arange(num_weapons)

##############################################################################
## Run DACOA and Compare to Brute Force

#Algorithm Parameters
delta = .01 #dual regularization parameter
rho = .0009 #delta/(delta ** 2 + 2)    #dual stepsize
gamma = .01 #primal stepsize

normDiff = [0]
evDiff = [0]


#PKs and V for Simplified 5v4 case:
# V=np.array([5, 6, 4, 5])

# Pk=np.array([[0.5, 0.3, 0.4, 0.3],
#                 #[.2, .6, .5, .4],
#                 [.6, .3, .4, .1],
#                 [.3, .2, .6, .7],
#                 [.4, .3, .7, .5]])
    
Pk= np.array([[0.8, 0.5, 0.1, 0.1],
              [0.6, 0.5, 0.1, 0.1],
              [0.2, 0.5, 0.1, 0.1],
              [0.3, 0.1, 0.7, 0.5],
              [0.3, 0.1, 0.7, 0.3]])
V=np.array( [ 7, 4, 2, 4 ])

# Pkorig= np.array([[0.7, 0.7, 0.2, 0.2, 0.1, 0.1],
#               [0.7, 0.7, 0.2, 0.2, 0.1, 0.1],
#               [0.7, 0.7, 0.2, 0.2, 0.1, 0.1],
#               [0.7, 0.7, 0.2, 0.2, 0.1, 0.1],
#               [0.7, 0.7, 0.2, 0.2, 0.1, 0.1],
#               [0.1, 0.1, 0.6 ,0.6, 0.2, 0.1],
#               [0.1, 0.1, 0.6 ,0.6, 0.2, 0.1],
#               [0.1, 0.1, 0.6 ,0.6, 0.2, 0.1],
#               [0.1, 0.1, 0.6 ,0.6, 0.2, 0.1],
#               [0.1, 0.1, 0.6 ,0.6, 0.2, 0.1],
#               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
#               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
#               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
#               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
#               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]])
# distAdj = np.array([[0.09, 0.01, 0.07, 0.06, 0.03, 0.03],
#                     [0.09, 0.07, 0.07, 0.04, 0.05, 0.1],
#                     [0.09, 0.1, 0.1, 0.07, 0.01, 0.06],
#                     [0.02, 0.03, 0.07, 0.07, 0.02, 0.1],
#                     [0.08, 0.08, 0.01, 0.05, 0.01, 0.09],
#                     [0.04, 0.01, 0.07, 0.04, 0.09, 0.02],
#                     [0.07, 0.06,	0.07, 0.1, 0.02, 0.01],
#                     [0.01, 0.07, 0.05, 0.01, 0.07, 0.04],
#                     [0.04, 0.08, 0.09, 0.06, 0.09, 0.09],
#                     [0.04, 0.03, 0.03, 0.07, 0.04, 0.05],
#                     [0.03, 0.07, 0.09, 0.1, 0.03, 0.01],
#                     [0.08, 0.1, 0.02, 0.02,	 0.1, 0.01],
#                     [0.01, 0.05, 0.03, 0.03, 0.01, 0.04],
#                     [0.1, 0.09, 0.03, 0.03,	0.07, 0.06],
#                     [0.07, 0.07, 0.08, 0.1, 0.06, 0.07]])

# Pk = Pkorig #- distAdj

# V = np.array([4,4,5,5,6,8])

for k in range(1,2):
    print("~~~")
    print(k)
    #Pk = np.random.random((5, 4))*0.25 + 0.5
    #V = np.reshape(np.random.random((1, 4))*5. + 5, (-1))
    
    
    inputs=WTAinputs(num_weapons, num_targets, Pk,V)
    comm =commClass(1)
    
    # #DACOA Setup
    # opt = DACOA(delta,gamma,rho, n, m, inputs, comm)
    # #opt.inputFiles("inputs_wta","WTA_inputs.communicate")
    # opt.setBlocks(pBlocks,np.arange(m))
    # opt.useScalars()
    # opt.setInit(.5*np.ones(n), np.zeros(m))
    # opt.stopIf(10e-4,5000,1)
    
    # #DACOA Run and Post-Processing
    # opt.run()
    # xUpdated = np.copy(opt.xFinal)
    # xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
    # #print(xUpdated)
    # selection = np.argmax(xUpdated, axis=1)
    # print("DACOA:")
    # print(selection)
    
    # xOrig = np.copy(xUpdated)
    # achieved_comp = np.ones((num_targets,1));
    # for i in range(num_weapons):
    #   achieved_comp[selection[i]] = achieved_comp[selection[i]]*inputs.Q[i, selection[i]]
    # valueOrig = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    # print(valueOrig)
    # print(xUpdated)
    
    
    #Brute Force Computation and Procession

    print("~~~")
    print("Opt:")
    EV = 100.
    index = 0.
    for i in range(int(pow(num_targets, num_weapons))):
      achieved_comp = np.ones((num_targets,1));
      remainder = i;
      for j in range(num_weapons):
        target = int(np.floor(remainder/pow(num_targets, num_weapons-j-1)));
    
        remainder = np.mod(remainder, pow(num_targets, num_weapons-j-1));
        #print("W: %d \t T: %d"%(j, target))
        achieved_comp[target] = achieved_comp[target]*inputs.Q[j, target];
    
      value = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    
      if value < EV:
        EV = value
        index = i
    actual = np.zeros((num_weapons,1)).astype(int)
    remainder = index
    for j in range(num_weapons):
      actual[j] = int(np.floor(remainder/pow(num_targets, num_weapons-j-1)));
      remainder = np.mod(remainder, pow(num_targets, num_weapons-j-1));
    print(np.reshape(actual,(-1)))
    print(EV)

    #############################################################################
    # Run DACOA with Regularization
    
    inputsReg=WTAinputsReg(num_weapons, num_targets, Pk, V)
    opt = DACOA(delta,gamma,rho, n, m, inputsReg, comm)
    opt.setBlocks(pBlocks,np.arange(m))
    opt.useScalars()
    opt.setInit(.5*np.ones(n),np.zeros(m))
    opt.stopIf(0,5000,1)
    
    #DACOA Run and Post-Processing
    opt.run()
    xUpdated = np.copy(opt.xFinal)
    xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
    #print(xUpdated)
    selection = np.argmax(xUpdated, axis=1)
    numIter = opt.numIter
    print("~~~")
    print("DACOA with Reg:")
    print(selection)
    print(xUpdated)


    achieved_comp = np.ones((num_targets,1));
    for i in range(num_weapons):
      achieved_comp[selection[i]] = achieved_comp[selection[i]]*inputs.Q[i, selection[i]]
    valueReg = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    print(valueReg)
    print(numIter)
    
    # x=.5*np.ones(n)
    # mu = np.zeros(m)
    # xSingle= []
    # Updated = np.zeros(num_targets*num_weapons)
    # Updated1 = np.zeros(num_targets*num_weapons)
    # muUpdated = np.zeros(num_weapons)
    # selection2 = [np.zeros(num_weapons)]
    
    # for ii in range(0,500):
    #     for jj in range(0,num_weapons):
    #         Updated[jj*num_targets:(jj+1)*num_targets] = opt.singlePrimal(x, mu, jj,inputsReg)[jj*num_targets:(jj+1)*num_targets]
    #     x=np.copy(Updated)   
    #     for jj in range(0,num_weapons):
    #         muUpdated[jj] = opt.singleDual(x,mu[jj],jj,inputs)
    #     mu = np.copy(muUpdated)
    #     xSingle.append(x)
    #     xUpdated2 = np.reshape(x, (num_weapons, num_targets))
    #     selection2.append(np.argmax(xUpdated, axis=1))
    
    # kk=1
    # xDiff=0
    # xDiffNorm = []
    # for jj in np.arange(1,numIter):
    #     xDiff = primals[jj][kk] - xSingle[jj-1][kk*num_targets:(kk+1)*num_targets]
    #     xDiffNorm.append(np.linalg.norm(xDiff))

primals=[]
selection=[]

for k in np.arange(opt.numIter+1):
    primals.append(np.reshape(opt.xValues[k], (num_weapons, num_targets)))
    selection.append(np.argmax(primals[k], axis=1))
    
#Plotting
weapon1 = [0]
weapon2 = [0]
weapon3 = [0]
weapon4 = [0]
weapon5 = [0]
weapon6 = [0]
weapon7 = [0]
weapon8 = [0]
weapon9 = [0]
weapon10 = [0]
weapon11 = [0]
weapon12 = [0]
weapon13 = [0]
weapon14 = [0]
weapon15 = [0]
for i in np.arange(1,opt.numIter+1):
    weapon1.append(selection[i][0])
    weapon2.append(selection[i][1])
    weapon3.append(selection[i][2])
    weapon4.append(selection[i][3])
    weapon5.append(selection[i][4])
    # weapon6.append(selection[i][5])
    # weapon7.append(selection[i][6])
    # weapon8.append(selection[i][7])
    # weapon9.append(selection[i][8])
    # weapon10.append(selection[i][9])
    # weapon11.append(selection[i][10])
    # weapon12.append(selection[i][11])
    # weapon13.append(selection[i][12])
    # weapon14.append(selection[i][13])
    # weapon15.append(selection[i][14])

plt.semilogy(np.arange(1,opt.numIter+1), opt.iterNorm[1:])
plt.ylabel("Distance Between Iterations")
plt.xlabel("Iteration Number")
plt.title("Convergence as a Function of Iteration Number")
plt.show()

plt.plot(np.arange(2,numIter+1), weapon1[2:], label="Weapon 1")
plt.plot(np.arange(2,numIter+1), weapon2[2:], label="Weapon 2")   
plt.plot(np.arange(2,numIter+1), weapon3[2:], label="Weapon 3")
plt.plot(np.arange(2,numIter+1), weapon4[2:], label="Weapon 4")
plt.plot(np.arange(2,numIter+1), weapon5[2:], label="Weapon 5")
# plt.plot(np.arange(2,numIter+1), weapon6[2:], label="Weapon 6")
# plt.plot(np.arange(2,numIter+1), weapon7[2:], label="Weapon 7")   
# plt.plot(np.arange(2,numIter+1), weapon8[2:], label="Weapon 8")
# plt.plot(np.arange(2,numIter+1), weapon9[2:], label="Weapon 9")
# plt.plot(np.arange(2,numIter+1), weapon10[2:], label="Weapon 10")
# plt.plot(np.arange(2,numIter+1), weapon11[2:], label="Weapon 11")
# plt.plot(np.arange(2,numIter+1), weapon12[2:], label="Weapon 12")   
# plt.plot(np.arange(2,numIter+1), weapon13[2:], label="Weapon 13")
# plt.plot(np.arange(2,numIter+1), weapon14[2:], label="Weapon 14")
# plt.plot(np.arange(2,numIter+1), weapon15[2:], label="Weapon 15")
#plt.legend()
plt.ylabel("Target Choice")
plt.xlabel("Iteration Number")
plt.title("Weapon-Target Assignments")
plt.show()

weapon1 = [0]
weapon2 = [0]
weapon3 = [0]
weapon4 = [0]
weapon5 = [0]
# weapon6 = [0]
# weapon7 = [0]
# weapon8 = [0]
# weapon9 = [0]
# weapon10 = [0]
# weapon11 = [0]
# weapon12 = [0]
# weapon13 = [0]
# weapon14 = [0]
# weapon15 = [0]
for i in np.arange(1,opt.numIter+1):
    weapon1.append(primals[i][0])
    weapon2.append(primals[i][1])
    weapon3.append(primals[i][2])
    weapon4.append(primals[i][3])
    weapon5.append(primals[i][4])
    # weapon6.append(selection[i][5])
    # weapon7.append(selection[i][6])
    # weapon8.append(selection[i][7])
    # weapon9.append(selection[i][8])
    # weapon10.append(selection[i][9])
    # weapon11.append(selection[i][10])
    # weapon12.append(selection[i][11])
    # weapon13.append(selection[i][12])
    # weapon14.append(selection[i][13])
    # weapon15.append(selection[i][14])

plt.plot(np.arange(2,numIter+1), weapon5[2:])
plt.ylabel("Target Choice")
plt.xlabel("Iteration Number")
plt.title("Weapon-Target Assignments")
plt.show()


##############################################################################
    ## Run DACOA with Path Dependent PKs
    
#Import regularized inputs file + Initialize Vectors
# inputsReg=WTAinputsReg(num_weapons, num_targets, Pk, V)
# comm =commClass(1)
# rate=.01
# fullRun = np.zeros(num_weapons)
# fullIter = 0
# weapon1 = [0]
# weapon2 = [0]
# weapon3 = [0]
# weapon4 = [0]
# weapon5 = [0]
# weapon6 = [0]
# weapon7 = [0]
# weapon8 = [0]
# weapon9 = [0]
# weapon10 = [0]
# weapon11 = [0]
# weapon12 = [0]
# weapon13 = [0]
# weapon14 = [0]
# weapon15 = [0]
# initP = .5*np.ones(n)
# initD = np.zeros(m)
# #DACOA Setup
# for k in np.arange(0,26):   #decide how many groups to run
#     inputsReg=WTAinputsReg(num_weapons, num_targets, Pk, V)
#     opt = DACOA(delta,gamma,rho, n, m, inputsReg, comm)
#     opt.setBlocks(pBlocks,np.arange(m))
#     opt.useScalars()
#     opt.setInit(initP,initD)
#     opt.stopIf(10e-4,100,1)
    
#     #DACOA Run and Post-Processing
#     opt.run()
#     initP = opt.xFinal
#     initD = opt.muFinal
#     xUpdated = np.copy(opt.xFinal)
#     xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
#     #print(xUpdated)
#     selection = np.argmax(xUpdated, axis=1)
#     print("~~~")
#     print("DACOA with Reg:")
#     print(k)
#     #print(selection)
#     print(opt.numIter)
#     for i in np.arange(num_weapons):
#         Pk[i][selection[i]] = Pk[i][selection[i]]+rate
#     #print(Pk)
#     for i in np.arange(1,opt.numIter+1):
#         weapon1.append(selection[i][0])
#         weapon2.append(selection[i][1])
#         weapon3.append(selection[i][2])
#         weapon4.append(selection[i][3])
#         weapon5.append(selection[i][4])
#         weapon6.append(selection[i][5])
#         weapon7.append(selection[i][6])
#         weapon8.append(selection[i][7])
#         weapon9.append(selection[i][8])
#         weapon10.append(selection[i][9])
#         weapon11.append(selection[i][10])
#         weapon12.append(selection[i][11])
#         weapon13.append(selection[i][12])
#         weapon14.append(selection[i][13])
#         weapon15.append(selection[i][14])
#     fullIter = fullIter + opt.numIter
    
# xReg = np.copy(xUpdated)
# print(fullIter)

# # Plotting
# plt.semilogy(np.arange(1,opt.numIter+1), opt.iterNorm[1:])
# plt.ylabel("Distance Between Iterations")
# plt.xlabel("Iteration Number")
# plt.title("Convergence as a Function of Iteration Number")
# plt.show()

# plt.plot(np.arange(2,fullIter+1), weapon1[2:], label="Weapon 1")
# plt.plot(np.arange(2,fullIter+1), weapon2[2:], label="Weapon 2")   
# plt.plot(np.arange(2,fullIter+1), weapon3[2:], label="Weapon 3")
# plt.plot(np.arange(2,fullIter+1), weapon4[2:], label="Weapon 4")
# plt.plot(np.arange(2,fullIter+1), weapon5[2:], label="Weapon 5")
# plt.plot(np.arange(2,fullIter+1), weapon6[2:], label="Weapon 1")
# plt.plot(np.arange(2,fullIter+1), weapon7[2:], label="Weapon 2")   
# plt.plot(np.arange(2,fullIter+1), weapon8[2:], label="Weapon 3")
# plt.plot(np.arange(2,fullIter+1), weapon9[2:], label="Weapon 4")
# plt.plot(np.arange(2,fullIter+1), weapon10[2:], label="Weapon 5")
# plt.plot(np.arange(2,fullIter+1), weapon11[2:], label="Weapon 1")
# plt.plot(np.arange(2,fullIter+1), weapon12[2:], label="Weapon 2")   
# plt.plot(np.arange(2,fullIter+1), weapon13[2:], label="Weapon 3")
# plt.plot(np.arange(2,fullIter+1), weapon14[2:], label="Weapon 4")
# plt.plot(np.arange(2,fullIter+1), weapon15[2:], label="Weapon 5")
# #plt.legend()
# plt.ylabel("Target Choice")
# plt.xlabel("Iteration Number")
# plt.title("Weapon-Target Assignments")
# plt.show()
