
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:42 2020

@author: kat.hendrickson
"""

import numpy as np
import scipy.linalg as la

class DACOA():
    def __init__(self, delta, gamma, rho, n, m, inputClass, commClass):
        """Initialize DACOA algorithm with inputs.
        Parameters:
            delta: dual regularization parameter
            gamma: primal stepsize
            rho: dual stepsize
            n: size of the primal variable
            m: size of the dual variable
            inputClass: input class that contains the gradPrimal, gradDual, projPrimal, and projDual functions
            commClass: input class that contains the comm function
        """
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
        self.tolerance = 10 ** -8   #updated with stopIf function
        self.maxIter = 10 ** 3      #updated with stopIf function
        self.flagIter=0     #updated with stopIf function
        self.xInit = np.zeros(n)
        self.muInit = np.zeros(m)
        self.scalarFlag=0
        self.commRate = 1
        self.inputClass = inputClass
        self.commClass = commClass

    def setActual(self,xActual,muActual):
        """If known, true values for primal and dual variables may be set
        and used for error calculations.
        
        Inputs:
            xActual - true value for primal variable
            muActual - true value for dual variable
        
        Outputs:
            If set, class values self.xError and self.muError will be calculated.
            These vectors calculate the error calculate the L2 norm of the
            distance between xActual and the iterate's value for the primal 
            variable."""
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
        
    ## Divide vectors into blocks larger than scalars
    def defBlocks(self,xBlocks,muBlocks):
        self.xBlocks = np.copy(xBlocks)
        self.xBlocks = np.append(self.xBlocks,[self.n])  #used to know the end of the last block.
        self.muBlocks = np.copy(muBlocks)
        self.muBlocks = np.append(self.muBlocks,[self.m])  #used to know the end of the last block.
    
    def setInit(self,xInit,muInit):
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
    
    ## Change Algorithm Stopping Parameters
    def stopIf(self,tol,maxIter,flagIter, suppress=1):
        self.tolerance = tol
        self.maxIter = maxIter
        self.flagIter = flagIter    #1 = stop based when maxIter reached
        if suppress == 0:
            print("Tolerance set to: ",self.tolerance)
            print("Max number of iterations set to: ",self.maxIter)
            print("Algorithm will stop when max number of iterations is reached: ", bool(self.flagIter))

    
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
        xError=[la.norm(self.xInit- self.xActual,2)]
        muError=[la.norm(mu-self.muActual,2)]
        gradRow = np.zeros(self.n)
        gradMatrix= np.array(gradRow)
        xVector = np.copy(self.xInit)
        
        # Convergence Parameters
        k=0
        while convdiff[k] > self.tolerance:
            
            if (self.flagIter == 1 and k >= self.maxIter):
                break
            
            prevIter = np.copy(xVector)   #used to determine convdiff
            
            # Update Primal Variables
            for p in range(Np):
                x = np.copy(Xp[:,p])
                a=self.xBlocks[p]  #lower boundary of block (included)
                b=self.xBlocks[p+1] #upper boundary of block (not included)
                if self.scalarFlag == 0:
                    pGradient = self.inputClass.gradPrimal(self,x,mu,p)
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
            [Xp, Xd, dup] = self.commClass.comm(self, Xp, Xd)
            
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
