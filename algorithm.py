
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:42 2020

@author: kat.hendrickson
"""

import numpy as np
import scipy.linalg as la

class DACOA():
    def __init__(self, delta, gamma, rho, n, m, inputClass, commClass, updateProb=1):
        """Initialize DACOA class. 
        Inputs:  
            * delta: dual regularization parameter  
            * gamma: primal stepsize  
            * rho: dual stepsize  
            * n: dimension of entire primal variable  
            * m: dimension of entire dual variable  
            * inputClass: input class that contains the gradPrimal, gradDual, projPrimal, and projDual functions  
            * commClass: input class that contains the comm function
            * updateProb: optional input that specifies the probability a primal agent performs an update at any given time k. Default value is 1. 
        """
        self.delta=delta
        self.gamma=gamma
        self.rho=rho
        self.n = n
        self.m = m 
        self.updateProb = updateProb
        
        #Default Values:
        self.xActual=np.zeros(n)
        """Value used to compute primal error; Set with `DACOA.setActual` method."""
        self.muActual=np.zeros(m)
        """Value used to compute dual error; Set with `DACOA.setActual` method."""
        self.flagActual=0   #used to determine whether to store error data later.
        self.xBlocks=np.arange(n+1)    
        """ Array that defines the dimensions of the primal blocks. Set with `DACOA.setBlocks` method."""
        self.muBlocks=np.arange(m+1)      
        """ Array that defines the dimensions of the dual blocks. Set with `DACOA.setBlocks` method."""
        self.tolerance = 10 ** -8   #updated with stopIf function
        self.maxIter = 10 ** 5      #updated with stopIf function
        self.maxIterBool=1     #updated with stopIf function
        self.xInit = np.zeros(n)
        """ Initial primal variable that is used at the start of optimization.
        The default value is a zero vector. Set with `DACOA.setInit` method. """
        self.muInit = np.zeros(m)
        """ Initial dual variable that is used at the start of optimization.
        The default value is a zero vector. Set with `DACOA.setInit` method. """
        self.scalarFlag=0
        self.inputClass = inputClass
        self.commClass = commClass

    def setActual(self,xActual,muActual):
        """If known, true values for primal (`DACOA.xActual`) and dual (`DACOA.muActual`) variables may be set
        and used for error calculations. If set, values `DACOA.xError` and `DACOA.muError` will be calculated.  
            These vectors calculate the L2 norm of the
            distance between the true value and the iterate's value for the primal and dual
            variables."""
        self.flagActual=1
        if np.size(xActual) == self.n:
            self.xActual=np.copy(xActual)
        else: 
            self.flagActual=0
            print("Error: Dimension mismatch between xActual and previously defined n.")
        if np.size(muActual) == self.m:
            self.muActual=np.copy(muActual)
        else:
            self.flagActual=0
            print("Error: Dimension mismatch between muActual and previously defined m.")
        
    def setBlocks(self,xBlocks,muBlocks):
        """ Defines the non-scalar primal and dual blocks. xBlocks is an array containing the first index for each primal block. 
        For example, with two primal agents, Agent 1's block always starts at 0 but Agent 2's block may start at entry 4. You'd then have the array xBlock = np.array([0,4]).  
        Similarly, muBlocks is an array containing the beginning indices for all dual agents."""
        self.xBlocks = np.copy(xBlocks)
        self.xBlocks = np.append(self.xBlocks,[self.n])  #used to know the end of the last block.
        self.muBlocks = np.copy(muBlocks)
        self.muBlocks = np.append(self.muBlocks,[self.m])  #used to know the end of the last block.
    
    def setInit(self,xInit,muInit):
        """Sets the initial values for the primal variable `DACOA.xInit` and the dual variable `DACOA.muInit`.  
        If this method is not used, zero vectors are used as a default."""
        if np.size(xInit) == self.n:
            self.xInit=np.copy(xInit)
        else:
            print("Error: Dimension mismatch between xInit and previously defined n.")
        if np.size(muInit) == self.m:
            self.muInit = np.copy(muInit)
        else:
            print("Error: Dimension mismatch between muInit and previously defined m.")
    
    def useScalars(self):
        self.scalarFlag=1
    

    def stopIf(self,tolerance,maxIter,maxIterBool=1):
        """Sets optimization stopping parameters by setting the following DACOA method values:
            * `DACOA.tolerance` : tolerance for distance between iterations. When this tolerance is reached, the optimization algorithm stops.  
            * `DACOA.maxIter` : max number of iterations to run. When this number of iterations is reached, the optimization algorithm stops.  
            * `DACOA.maxIterBool` : optional boolean input that determines whether the optimization code always stops when `DACOA.maxIter` is reached (1) 
            or whether it ignores  `DACOA.maxIter` and instead runs until  `DACOA.tolerance` is reached (0). Default value is (1)."""
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.maxIterBool = maxIterBool    #1 = stop based when maxIter reached

    def run(self):

        # Initialize Primal and Dual Variables
        Np = np.size(self.xBlocks)-1  #number of primal agents
        Nd = np.size(self.muBlocks)-1 #number of dual agents
        self.Np = Np
        self.Nd = Nd
        Xp = np.outer(self.xInit,np.ones(self.Np))  #initialize primal matrix
        Xd = np.outer(self.xInit,np.ones(self.Nd))       #initialize dual matrix
        mu = np.copy(self.muInit)
    
        dCount = np.zeros((Np,Nd))    #initialize dCount vector
        t = np.zeros(Nd)    #initialize t vector for each dual agent
        convdiff=[self.tolerance + 100]        #initialize convergence distance measure
        #xError=[la.norm(self.xInit- self.xActual,2)]
        #muError=[la.norm(mu-self.muActual,2)]
        gradRow = np.zeros(self.n)
        gradMatrix= np.array(gradRow)
        xVector = np.copy(self.xInit)
        xValues = [np.zeros(self.n)]
        
        
        # Convergence Parameters
        k=0
        while convdiff[k] > self.tolerance:
            
            # if (k % 500 == 0):
            #     print(k, "iterations...")
            
            if (self.maxIterBool == 1 and k >= self.maxIter):
                break
            
            #prevIter = np.copy(xVector)   #used to determine convdiff
            # Update Primal Variables
            for p in range(Np):
                x = np.copy(Xp[:,p])
                a=self.xBlocks[p]  #lower boundary of block (included)
                b=self.xBlocks[p+1] #upper boundary of block (not included)
                updateDraw = np.random.rand(1)
                if self.scalarFlag == 0:
                    if updateDraw <= self.updateProb:
                        pGradient = self.inputClass.gradPrimal(self,x,mu,p)
                    else:
                        pGradient = np.zeros(b-a)
                    gradRow[a:b]=pGradient
                    pUpdate = x[a:b] - self.gamma*pGradient
                    Xp[a:b,p] = self.inputClass.projPrimal(pUpdate)
                    xVector[a:b] = np.copy(Xp[a:b,p])
                elif self.scalarFlag == 1:
                    for i in range(a,b):
                        pGradient = self.inputClass.gradPrimal(self,x,mu,i)
                        gradRow[i]=pGradient
                        pUpdate = x[i] - self.gamma*pGradient
                        Xp[i,p] = self.inputClass.projPrimal(pUpdate)
                        xVector[i] = np.copy(Xp[i,p])
            gradMatrix =np.vstack((gradMatrix,gradRow))
           
            
            # Communicate Primal Updates
            #dup=0
            #if k in 50*np.arange(0,101):
            [Xp, Xd, dup] = self.commClass.comm(self, Xp, Xd)
                #print("Communicated at:")
                #print(k)
            
            # Update Dual Variables if they have received updates from all primal agents
            dCount = dCount + dup
            dCount_nz = np.count_nonzero(dCount, axis=0)
            for d in range(Nd):
                if dCount_nz[d] >= Np:
                    a=self.muBlocks[d]  #lower boundary of block (included)
                    b=self.muBlocks[d+1] #upper boundary of block (not included)
                    if self.scalarFlag == 0:
                        dGradient = self.inputClass.gradDual(self,Xd[:,d],mu[a:b],d)
                        dUpdate = mu[a:b] + self.rho*dGradient
                        t[d]=t[d]+1
                        mu[a:b] = self.inputClass.projDual(dUpdate)
                    elif self.scalarFlag == 1:
                        muNew = np.copy(mu)  #stored separately so block updates don't use new data
                        for j in range(a,b):
                            dGradient = self.inputClass.gradDual(self,Xd[:,d],mu[j],j)
                            dUpdate = mu[j] + self.rho*dGradient
                            t[d]=t[d]+1
                            muNew[j] = self.inputClass.projDual(dUpdate)
                        mu = np.copy(muNew)
                    dCount=np.zeros((Np,Nd))    # resets update counter for ALL dual agents if at least one has updated
            # Calculate Iteration Distance and Errors
            k=k+1       # used to count number of iterations (may also be used to limit runs)
            newIter = np.copy(xVector)
            xValues.append(newIter)
            #iterNorm = la.norm(prevIter - newIter)  #L2 norm of the diff between prev and current iterations
            convdiff.append(1)
            #xError.append(la.norm(xVector - self.xActual,2))
            #muError.append(la.norm(mu-self.muActual,2))
        
        #if self.flagActual == 1:
            #self.xError = xError
            #self.muError = muError
        
        self.iterNorm = 1
        self.numIter = k
        self.xFinal=xVector
        self.muFinal = mu
        self.gradMatrix = gradMatrix
        self.xValues = xValues
        return self.xFinal, self.muFinal
    
    def singlePrimal(self,x,mu,agent,inputClass):
        xUpdated = np.copy(x)
        a=self.xBlocks[agent]  #lower boundary of block (included)
        b=self.xBlocks[agent+1] #upper boundary of block (not included)
        
        if self.scalarFlag == 0:
            pGradient = inputClass.gradPrimal(self,x,mu,agent)
            pUpdate = x[a:b] - self.gamma*pGradient
            xUpdated[a:b] = inputClass.projPrimal(pUpdate)
        elif self.scalarFlag == 1:
            for i in range(a,b):
                pGradient = inputClass.gradPrimal(self,x,mu,i)
                pUpdate = x[i] - self.gamma*pGradient
                # print(pUpdate)
                xUpdated[i] = inputClass.projPrimal(pUpdate)
        return xUpdated

    def singleDual(self,x,mu,agent,inputClass):
        """ Note: This assumes that the check for primal updates has been handled elsewhere.
        
        The mu here is only that agents block - not the entirety of the mu vector. This is because dual agents need not receive other dual updates."""
        
        muUpdated=np.copy(mu)
        a=self.muBlocks[agent]  #lower boundary of block (included)
        b=self.muBlocks[agent+1]
        i=0
        
        if self.scalarFlag == 0:
            dGradient = inputClass.gradDual(self,x,mu,agent)
            dUpdate = mu + self.rho*dGradient
            muUpdated = inputClass.projDual(dUpdate)
        elif self.scalarFlag == 1:
            if np.size(mu) == 1:
                dGradient = inputClass.gradDual(self,x,mu,agent)
                dUpdate = mu + self.rho*dGradient
                muUpdated = inputClass.projDual(dUpdate)
            else:
                for j in range(a,b):
                    dGradient = inputClass.gradDual(self,x,mu[i],j)
                    dUpdate = mu[i] + self.rho*dGradient
                    muUpdated[i] = inputClass.projDual(dUpdate)
                    i += 1
        
        return muUpdated
    

