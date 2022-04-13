#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from algorithm import DACOA
import sys
import os
sys.path.append('./WTA_inputs')
# Import inputs file.
from WTA_inputs.inputs_wta import WTAinputs
from WTA_inputs.inputs_wtaReg import WTAinputsReg
from WTA_inputs.communicate import commClass
import csv

with open('./ran_multiple_times.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(['Optimal Assignment', 'Optimal EV', 'DACOA Assignment', 'DACOA EV'])

num_weapons = 5
num_targets = 4
n = num_weapons * num_targets    #size of primal variable
m = num_weapons                       #size of dual variable
pBlocks = num_targets*np.arange(num_weapons)

#Algorithm Parameters
delta = .01 #dual regularization parameter
rho = .0009 #delta/(delta ** 2 + 2)    #dual stepsize
gamma = .1 #primal stepsize

normDiff = [0]
evDiff = [0]
number_of_time_you_want_to_run = 50
error=np.zeros(number_of_time_you_want_to_run)
numIter = np.zeros(number_of_time_you_want_to_run)

for k in range(0,number_of_time_you_want_to_run):
    Pk = np.random.uniform(low=0.2, high=0.9, size=(num_weapons, num_targets))
    V = np.random.uniform(low=2, high=10, size=(num_targets))

    inputs=WTAinputs(num_weapons, num_targets, Pk,V)
    comm =commClass(1)

    EV = 100.
    index = 0.
    for i in range(int(pow(num_targets, num_weapons))):
        achieved_comp = np.ones((num_targets, 1))
        remainder = i
        for j in range(num_weapons):
            target = int(np.floor(remainder / pow(num_targets, num_weapons - j - 1)))

            remainder = np.mod(remainder, pow(num_targets, num_weapons - j - 1))
            # print("W: %d \t T: %d"%(j, target))
            achieved_comp[target] = achieved_comp[target] * inputs.Q[j, target]

        value = np.dot(np.reshape(achieved_comp, -1), np.reshape(inputs.V, -1))

        if value < EV:
            EV = value
            index = i
    actual = np.zeros((num_weapons, 1)).astype(int)
    remainder = index
    for j in range(num_weapons):
        actual[j] = int(np.floor(remainder / pow(num_targets, num_weapons - j - 1)))
        remainder = np.mod(remainder, pow(num_targets, num_weapons - j - 1))

    # print(np.reshape(actual,(-1)))
    # print(EV)
    # print(Pk)
    # print(V)

    inputsReg=WTAinputsReg(num_weapons, num_targets, Pk, V)
    opt = DACOA(delta,gamma,rho, n, m, inputsReg, comm)
    opt.setBlocks(pBlocks,np.arange(m))
    opt.useScalars()
    opt.setInit((1./num_targets)*np.ones(n),np.zeros(m))
    opt.stopIf(-0.10,10000,1)
    #DACOA Run and Post-Processing
    opt.run()
    xUpdated = np.copy(opt.xFinal)
    xUpdated = np.reshape(xUpdated, (num_weapons, num_targets))
    #print(xUpdated)
    selection = np.argmax(xUpdated, axis=1)
    numIter[k] = opt.numIter
    # print("~~~")
    # print("DACOA with Reg:")
    # print(selection)
    # print(xUpdated)


    achieved_comp = np.ones((num_targets,1));
    for i in range(num_weapons):
      achieved_comp[selection[i]] = achieved_comp[selection[i]]*inputs.Q[i, selection[i]]
    valueReg = np.dot(np.reshape(achieved_comp,-1), np.reshape(inputs.V, -1));
    # print(valueReg)
    # # print(numIter)
    error[k] = (valueReg-EV)/EV
    row = [np.reshape(actual,(-1)), EV, selection, valueReg, error]
    
    

    with open('./ran_multiple_times.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(k,error[k])

print(np.mean(error))

