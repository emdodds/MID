# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:52:02 2016

@author: Eric
"""

import numpy as np
#import strfanalysis
from scipy.optimize import minimize

class MID:
    """A class to find the maximally informative stimulus dimensions for given stimuli and spikes. 
    Currently only finds the single most informative dimension."""
    
    def __init__(self, handler=None, nbins=15):
        """Input: handler, an object with a generator() method that returns an iterator
        over stim, spikecount pairs. handler also needs a stimshape attribute."""
        if handler is None:
            self.handler = strfanalysis.STRFAnalyzer()
        else:
            self.handler = handler
            
        self.v = self.vec_init('sta')
        self.nbins = nbins
        self.binbounds = self.decide_bins(nbins)
        
    def vec_init(self, method='random'):
        """If random, return a random normalized vector. Otherwise return a random normalized stimulus."""
        try:
            if method =='random':
                length = np.prod(self.handler.stimshape)
                vec = np.random.randn(length)
            elif method=='sta':
                vec = self.handler.get_STA() - self.handler.get_stimave()
            else:
                vec = self.handler.rand_stimresp(1)[0][0]
        except AttributeError:
            print('Specialized initialization failed. Falling back on first stimulus.')
            vec = self.handler.generator().__next__()[0]
        vec = vec/np.linalg.norm(vec)
        return vec
            
    def decide_bins(self, nbins = None, edgefrac = None):
        if nbins is None:
            nbins=self.nbins
        if edgefrac is None:
            edgefrac = 1/(5*nbins)
#        stims = self.handler.rand_stimresp(1000)[0]
#        projections = stims.dot(self.v)
        projections = np.zeros(self.handler.get_nstim())
        ii=0
        for stim,_ in self.handler.generator():
            projections[ii]=self.v.dot(stim)
            ii+=1
        projections = np.sort(projections)
        bottomind = int(len(projections)*edgefrac/2)
        topind= len(projections) - bottomind
        bottom = projections[bottomind]
        top = projections[topind]
        self.binbounds =  np.linspace(bottom, top, nbins)
        return self.binbounds
            
    def bin_ind(self, val):
        """Returns the index of the bin of projection values into which val falls."""    
        for ind in range(len(self.binbounds)):
            if val < self.binbounds[ind]:
                return ind
        return ind
    
    def info_and_dists(self,v=None, neg=True):
        """Returns the mutual information between spike arrival times and the projection along v."""
        if v is None:
            v = self.v
        self.decide_bins() 
        
        pv = np.zeros(self.nbins) # prob dist of projections
        pvt = np.zeros_like(pv) # prob dist of projections given spike (t for trigger)
        for stim, sp in self.handler.generator():
            proj = self.v.dot(stim)
            projbin = self.bin_ind(proj)
            pv[projbin] = pv[projbin] + 1
            pvt[projbin] = pvt[projbin] + sp
        
        pv = pv/np.sum(pv)
        pvt = pvt/np.sum(pvt)
        safepv = np.copy(pv)
        safepv[safepv==0] = np.min(safepv[np.nonzero(safepv)]) # prevents divide by zero errors when 0/0 below
        info = 0 
        for ii in range(len(pvt)):
            info += (0 if pvt[ii] == 0 else pvt[ii]*np.log2(pvt[ii]/safepv[ii]))
        info = info*(self.binbounds[1]-self.binbounds[0]) # units of dx
        factor = -1 if neg else 1
        return factor*info, pv, pvt
    
    def info(self, v=None, neg=True):
        return self.info_and_dists(v,neg)[0]
    
    def info_grad(self, v, neg=True):
        """Return the information as in infov, and the gradient of the same with respect to v.
        If neg, returns minus these things."""
        self.decide_bins()        
        
        pv = np.zeros(self.nbins) # prob dist of projections
        pvt = np.zeros_like(pv) # prob dist of projections given spike (t for trigger)
        sv = np.zeros((self.nbins,len(v))) # mean stim given projection
        svt = np.zeros_like(sv) # mean stim given projection and spike
        nstims = 0
        nspikes = 0
        for stim, sp in self.handler.generator():
            proj = v.dot(stim)
            projbin = self.bin_ind(proj)
            pv[projbin] = pv[projbin] + 1
            pvt[projbin] = pvt[projbin] + sp
            sv[projbin] = sv[projbin] + stim
            svt[projbin] = svt[projbin] + sp*stim
        
        
        nstims = np.sum(pv)
        nspikes = np.sum(pvt)
        pv = pv/nstims
        pvt = pvt/nspikes
        
        # to avoid dividing by zero I make zeros equal the next smallest possible value, which may cause problems if there are a lot of zeros
        safepv = np.copy(pv)
        safepv[safepv==0] = 1./nstims
        sv = (sv/nstims)/safepv[:,np.newaxis]
        safepvt = np.copy(pvt)
        safepvt[safepvt==0] = 1./nspikes
        svt = (svt/nspikes)/safepvt[:,np.newaxis]
        
        # Compute the derivative of the probability ratio wrt bin. 
        # This is approximating an integral over bins so the size of the bins doesn't enter the calculation
        deriv = np.gradient(pvt/safepv) # uses 2nd order method
        
        grad = np.sum(pv[:,np.newaxis]*(svt-sv)*deriv[:,np.newaxis],0)
        info = 0
        for ii in range(len(pvt)):
            info += (0 if pvt[ii] == 0 else pvt[ii]*np.log2(pvt[ii]/pv[ii]))
        info = info*(self.binbounds[1]-self.binbounds[0]) # units of dx
        factor = -1 if neg else 1
        return factor*info, factor*grad
    
    
    def grad_ascent(self, v, rate, gtol=1e-6, maxiter=100):
        gnorm = 2*gtol
        it=0
        infohist=[]
        print('Info             Gradient norm')
        while gnorm>gtol and it<maxiter:
            self.decide_bins()
            info, grad = self.info_grad(v,neg=False)
#            if it>0 and info<infohist[-1]:
#                print("Information not increasing. Reducing rate.")
#                rate=rate/2
#                it+=1 # iteration still counts
#                continue 
            infohist.append(info)
            gnorm = np.linalg.norm(grad)
            print(str(info)+'  '+str(gnorm))
            v = v + rate*grad 
            it+=1
        print(str(info)+'  '+str(gnorm))
        if gnorm<gtol:
            mes = "Converged to desired precision."
        else:
            mes = "Did not converge to desired precision."
        return SimpleResult(v, mes, history=infohist)
            
        
    def optimize(self, method='BFGS', rate=1e-6, maxiter=100):
        if method == 'BFGS':
            result = minimize(self.info_grad,self.v,method=method, jac=True, options={'disp':True, 'maxiter':maxiter})
        elif method == 'Nelder-Mead':
            result = minimize(self.info, self.v, method=method, options={'disp':True})
        elif method == 'GA':
            result = self.grad_ascent(self.v,rate, maxiter=maxiter)
        else:
            return SimpleResult(self.v, "No valid method provided. Did nothing.")
        
        print(result.message)
        self.v = result.x
        return result
        
class SimpleResult:
    def __init__(self, x, message, **kwargs):
        self.x = x
        self.message = message
        for key, val in kwargs.items():
            setattr(self, key, val)
        