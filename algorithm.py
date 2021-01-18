#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:42 2020

@author: kat.hendrickson
"""

import numpy as np
import scipy.linalg as la
import importlib

class DACOA():
    def __init__(self,delta, gamma, rho, n, m):
        self.delta=delta
        self.gamma=gamma
        self.rho=rho
        self.n = n      # dimension of entire primal vector
        self.m = m      # dimension of entire dual vector
        
        #Default Values:
        self.xActual=np.zeros(n)  #can be updated with setActual function
        self.muActual=np.zeros(m) #can be updated with setActual function
        self.flagActual=0   #sets flag to zero - used to determine whether to store error data later.
        self.xBlocks=np.arange(n+1)    #vector blocks are set with defBlocks
        self.muBlocks=np.arange(m+1)      #vector blocks are set with defBlocks
        self.filenames = ['inputs','communicate']
        self.tolerance = 10 ** -8   #updated with stopIf function
        self.maxIter = 10 ** 3      #updated with stopIf function
        self.flagIter=0     #updated with stopIf function
        self.xInit = np.zeros(n)
        self.muInit = np.zeros(m)

    def setActual(self,xActual,muActual):
        """Set the true primal and dual variables. If set, values will be used to calculate error.
        
        Inputs:
            xActual - true value for primal variable
            muActual - true value for dual variable"""
        self.flagActual=1
        if np.size(xActual) == self.n:
            self.xActual=xActual
        else: 
            self.flagActual=0
            print("Error: Dimension mismatch between xActual and previously defined n.")
        if np.size(muActual) == self.m:
            self.muActual=muActual
        else:
            self.flagActual=0
            print("Error: Dimension mismatch between muActual and previously defined m.")
        
    ## Divide vectors into blocks larger than scalars
    def defBlocks(self,xBlocks,muBlocks):
        self.xBlocks = xBlocks
        self.xBlocks = np.append(self.xBlocks,[self.n])  #used to know the end of the last block.
        self.muBlocks = muBlocks
        self.muBlocks = np.append(self.muBlocks,[self.m])  #used to know the end of the last block.
        
    ## Change input file names
    def inputFiles(self,newInputs,newComms):
        if isinstance(newInputs,str):
            self.filenames[0]=newInputs
        else:
            print("Error: Input filename must be a string.")
        if isinstance(newComms,str):
            self.filenames[1]=newComms
        else:
            print("Error: Communications filename must be a string.")
    
    def setInit(self,xInit,muInit):
        if np.size(xInit) == self.n:
            self.xInit=xInit
        else:
            print("Error: Dimension mismatch between xInit and previously defined n.")
        if np.size(muInit) == self.m:
            self.muInit = muInit
        else:
            print("Error: Dimension mismatch between muInit and previously defined m.")
    
    ## Change Algorithm Stopping Parameters
    def stopIf(self,tol,maxIter,flagIter):
        self.tolerance = tol
        self.maxIter = maxIter
        self.flagIter = flagIter    #1 = stop based when maxIter reached
        print("Tolerance set to: ",self.tolerance)
        print("Max number of iterations set to: ",self.maxIter)
        print("Algorithm will stop when max number of iterations is reached: ", bool(self.flagIter))
    
    def run(self):
        inputs = importlib.import_module(self.filenames[0])
        communicate = importlib.import_module(self.filenames[1])
        
        # Initialize Primal and Dual Variables
        Np = np.size(self.xBlocks)-1  #number of primal agents
        Nd = np.size(self.muBlocks)-1 #number of dual agents
        self.Np = Np
        self.Nd = Nd
        Xp = np.outer(self.xInit,np.ones(self.Np))  #initialize primal matrix
        Xd = np.outer(self.xInit,np.ones(self.Nd))       #initialize dual matrix
        mu = self.muInit
    
        dCount = np.zeros((Np,Nd))    #initialize dCount vector
        t = np.zeros(Nd)    #initialize t vector for each dual agent
        convdiff=[self.tolerance + 100]        #initialize convergence distance measure
        xError=[la.norm(np.diag(Xp) - self.xActual,2)]
        muError=[la.norm(mu-self.muActual,2)]
        gradRow = np.zeros(self.n)
        gradMatrix= np.array(gradRow)
        xVector = self.xInit
        
        # Convergence Parameters
        k=0
        while convdiff[k] > self.tolerance:
            
            if (self.flagIter == 1 and k >= self.maxIter):
                break
            
            prevIter = np.array(xVector)   #used to determine convdiff
            
            # Update Primal Variables
            for p in range(Np):
                x = Xp[:,p]
                a=self.xBlocks[p]  #lower boundary of block (included)
                b=self.xBlocks[p+1] #upper boundary of block (not included)
                pGradient = inputs.gradPrimal(self,x,mu,p)
                gradRow[a:b]=pGradient
                pUpdate = x[a:b] - self.gamma*pGradient
                Xp[a:b,p] = inputs.projPrimal(pUpdate)
                xVector[a:b] = Xp[a:b,p]
            gradMatrix =np.vstack((gradMatrix,gradRow))
            
            # Communicate Primal Updates
            [Xp, Xd, dup] = communicate.comm(self,Xp, Xd)
            
            # Update Dual Variables if they have received updates from all primal agents
            dCount = dCount + dup
            dCount_nz = np.count_nonzero(dCount, axis=0)
            for d in range(Nd):
                if dCount_nz[d] >= Np:
                    a=self.muBlocks[d]  #lower boundary of block (included)
                    b=self.muBlocks[d+1] #upper boundary of block (not included)
                    dGradient = inputs.gradDual(self,Xd[:,d],mu[a:b],d)
                    dUpdate = mu[a:b] + self.rho*dGradient
                    t[d]=t[d]+1
                    mu[a:b] = inputs.projDual(dUpdate)
                    dCount=np.zeros((Np,Nd))    # resets update counter for ALL dual agents if at least one has updated
            # Calculate Iteration Distance and Errors
            k=k+1       # used to count number of iterations (may also be used to limit runs)
            newIter = xVector
            iterNorm = la.norm(prevIter - newIter)  #L2 norm of the diff between prev and current iterations
            convdiff.append(iterNorm)
            xError.append(la.norm(xVector - self.xActual,2))
            muError.append(la.norm(mu-self.muActual,2))
        
        if self.flagActual == 1:
            self.xError = xError
            self.muError = muError
        
        self.iterNorm = convdiff
        self.numIter = k
        self.xFinal=xVector
        self.muFinal = mu
        self.gradMatrix = gradMatrix
        return self.xFinal, self.muFinal
