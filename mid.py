# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:52:02 2016

@author: Eric
"""

import numpy as np
from scipy.optimize import minimize
import os
cwd = os.getcwd()
strfandir = '../../cell attached analyzed 2015/'
os.chdir(strfandir)
import strfanalysis
os.chdir(cwd)

class MID:
    """A class to find the maximally informative stimulus dimensions for given stimuli and spikes. 
    Currently only finds the single most informative dimension."""
    
    def __init__(self, handler=None, nbins=15):
        """Input: handler, an object with a generator() method that returns an iterator
        over stim, spikecount pairs. handler also needs a stimshape attribute."""
        if handler is None:
            try:
                os.chdir(strfandir)
            except:
                pass
            self.handler = strfanalysis.STRFAnalyzer()
        else:
            self.handler = handler
            
        self.v = self.vec_init('sta')
        self.nbins = nbins
        self.binbounds = self.decide_bins(nbins=nbins)
        
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
        return np.array(vec)
            
    def decide_bins(self, v=None, nbins = None, edgefrac = None):
        if nbins is None:
            nbins=self.nbins
        if edgefrac is None:
            edgefrac = 1/(5*nbins)
        if v is None:
            v = self.v
#        stims = self.handler.rand_stimresp(1000)[0]
#        projections = stims.dot(self.v)
        projections = np.zeros(self.handler.get_nstim())
        ii=0
        for stim,_ in self.handler.generator():
            projections[ii]=v.dot(stim)
            ii+=1
        projections = np.sort(projections)
        bottomind = int(len(projections)*edgefrac/2)
        topind= len(projections) - bottomind
        bottom = projections[bottomind]
        top = projections[topind]
        self.binbounds =  np.linspace(bottom, top, nbins-1)
        
        return self.binbounds
            
    def bin_ind(self, val):
        """Returns the index of the bin of projection values into which val falls."""    
        for ind in range(len(self.binbounds)):
            if val < self.binbounds[ind]:
                return ind
        return ind+1
    
    def info_and_dists(self,v=None, neg=True):
        """Returns the mutual information between spike arrival times and the projection along v."""
        if v is None:
            v = self.v
        self.decide_bins(v=v) 
        
        pv = np.zeros(self.nbins) # prob dist of projections
        pvt = np.zeros_like(pv) # prob dist of projections given spike (t for trigger)
        for stim, sp in self.handler.generator():
            proj = v.dot(stim)
            projbin = self.bin_ind(proj)
            pv[projbin] = pv[projbin] + 1
            pvt[projbin] = pvt[projbin] + sp
        
        nstims = np.sum(pv)
        nspikes = np.sum(pvt)
        pv = pv/nstims
        pvt = pvt/nspikes
        safepv = np.copy(pv)
        safepv[safepv==0] = 1./nstims # prevents divide by zero errors when 0/0 below
        info = 0 
        for ii in range(len(pvt)):
            info += (0 if pvt[ii] == 0 else pvt[ii]*np.log2(pvt[ii]/safepv[ii]))
        factor = -1 if neg else 1
        return factor*info, pv, pvt
    
    def info(self, v=None, neg=True):
        return self.info_and_dists(v,neg)[0]
        
    def info_est(self):
        """Returns an estimate of the info/spike. Possibly a bad estimate if each stimulus is seen only once.
        Currently only works for STRFAnalyzer handler or equivalent."""
        # TODO: test
        Ispike = 0
        for name in np.unique(self.handler.namelist):
            inds = np.where(np.array(self.handler.namelist) == name)[0]
            combtrain = np.zeros(np.max(self.handler.triallengths[inds])) # assuming triallengths are all about the same
            for ii in inds:
                combtrain = combtrain + self.handler.spiketrain(ii)
            combtrain = combtrain/(inds.shape[-1])
            for prob in combtrain:
                if prob>0:
                    Ispike = Ispike + prob*np.log2(prob*self.handler.nstim/self.handler.nspikes)
        return Ispike/self.handler.nspikes
    
    def info_grad(self, v, neg=True):
        """Return the information as in infov, and the gradient of the same with respect to v.
        If neg, returns minus these things."""
        self.decide_bins(v=v)        
        
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
        factor = -1 if neg else 1
        return factor*info, factor*grad
    
    
    def grad_ascent(self, v, rate, gtol=1e-6, maxiter=100):
        gnorm = 2*gtol
        it=0
        infohist=[]
        print('Info             Gradient norm')
        for it in range(maxiter):
            info, grad = self.info_grad(v,neg=False)
#            if it>0 and info<infohist[-1]:
#                print("Information not increasing. Reducing rate.")
#                rate=rate/2
#                it+=1 # iteration still counts
#                continue 
            infohist.append(info)
            gnorm = np.linalg.norm(grad)
            if gnorm<gtol:
                break
            print(str(info)+'  '+str(gnorm))
            v = v + rate*grad 
        print(str(info)+'  '+str(gnorm))
        if gnorm<gtol:
            mes = "Converged to desired precision."
        else:
            mes = "Did not converge to desired precision."
        return SimpleResult(v, mes, history=infohist)
        
    def line_max_backtrack(self, v, initinfo, grad, params=None):
        if params is None:
            params = BacktrackingParams()
        bestinfo=initinfo
        step = params.maxstep
        beststep = 0
        goodenough = np.linalg.norm(grad)*params.acceptable
        for it in range(params.maxiter):
            newinfo = self.info(v+step*grad, neg=False)
            if newinfo-initinfo > goodenough*step:
                print("Satisficed with step size " + str(step), " on iteration " + str(it))
                return step
            if newinfo > bestinfo:
                bestinfo = newinfo
                beststep = step
                print("Found new best step size " + str(beststep) + " with info " + str(bestinfo))
            step = step*params.reducefactor
        print("Failed to satisfice. Updating with best found step size " + str(beststep))
        return beststep
    
    def GA_with_linemax(self, v, gtol=1e-5, maxiter=100, params=None):
        gnorm = 2*gtol
        it=0
        infohist=[]
        print('Info             Gradient norm')
        for it in range(maxiter):
            info, grad = self.info_grad(v,neg=False)
            assert infohist==[] or info > infohist[-1]
            infohist.append(info)
            gnorm = np.linalg.norm(grad)
            if gnorm < gtol:
                break
            print(str(info)+'  '+str(gnorm))
            step = self.line_max_backtrack(v, info, grad, params)
            if step == 0:
                print("No improvement found in direction of gradient.")
                break
            v = v + step*grad
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
        elif method == 'GA_with_linemax':
            result = self.GA_with_linemax(self.v, maxiter=maxiter)
        else:
            return SimpleResult(self.v, "No valid method provided. Did nothing.")
        
        print(result.message)
        self.v = result.x
        return result
        
#class AnnealingParameters:
#    def __init__(self, maxiter=1, Tinit = 0.01, Tfinal=1e-5, Tdownfactor=.95, Tupfactor=5, tolerance=5e-5, updatefactor=100):
#        self.maxiter=maxiter
#        self.Tinit=Tinit
#        self.Tfinal = Tfinal
#        self.Tdownfactor=Tdownfactor
#        self.Tupfactor=Tupfactor
#        self.tolerance = tolerance
#        self.updatefactor=updatefactor
    
class BacktrackingParams:
    def __init__(self, maxiter=10, maxstep=1, reducefactor=.5, acceptable=.5):
        self.maxiter = maxiter
        self.maxstep = maxstep
        self.reducefactor = reducefactor
        self.acceptable = acceptable
                
        
class SimpleResult:
    def __init__(self, x, message, **kwargs):
        self.x = x
        self.message = message
        for key, val in kwargs.items():
            setattr(self, key, val)
        