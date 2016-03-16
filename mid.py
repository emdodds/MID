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
    """A class to find the maximally informative stimulus dimensions for given stimuli and spikes."""
    
    def __init__(self, handler=None, ndim=1, nbins=15):
        """Input: handler, an object with a generator() method that returns an iterator
        over (stim, spikecount) pairs. handler also needs a stimshape attribute."""
        if handler is None:
            try:
                os.chdir(strfandir)
            except:
                pass
            self.handler = strfanalysis.STRFAnalyzer()
        else:
            self.handler = handler
            
        self.ndim = ndim
        self.vecs = np.zeros(self.ndim, np.prod(self.handler.stimshape))
        for d in self.ndim:
            self.vecs[d] = self.vec_init()
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
            print('Specialized initialization failed. Falling back on random early stimulus.')
            stims = np.zeros(1000, np.prod(self.handler.stimshape))
            index = 0
            for stim in self.handler.generator():
                if index>=1000:
                    break
                stims[index] = stim
                index+=1
            vec = stims[np.random.choice(1000)]
        vec = vec/np.linalg.norm(vec)
        return np.array(vec)
            
    def decide_bins(self, vecs=None, nbins = None, edgefrac = None):
        if nbins is None:
            nbins=self.nbins
        if edgefrac is None:
            edgefrac = 1/(5*nbins)
        if vecs is None:
            vecs = self.vecs
        nstim = self.handler.get_nstim()
        projections = np.zeros((self.ndim,nstim))
        ii=0
        for stim,_ in self.handler.generator():
            projections[:,ii]=vecs.dot(stim)
            ii+=1
        projections = np.sort(projections,0)
        bottomind = int(nstim*edgefrac/2)
        topind = nstim - bottomind
        bottoms = projections[:,bottomind]
        tops = projections[:,topind]
        self.binbounds =  np.zeros((self.ndim, self.nbins))
        for d in self.ndim:
            self.binbounds[d] = np.linspace(bottoms[d], tops[d], nbins-1)        
        return self.binbounds
            
    def bin_ind(self, val, dim=0):
        """Returns the index of the bin of projection values into which val falls."""    
        for ind in range(len(self.binbounds[dim])):
            if val < self.binbounds[dim,ind]:
                return ind
        return ind+1

    def info_est(self):
        """Returns an estimate of the info/spike. Possibly a bad estimate if each stimulus is seen only once.
        Currently only works for STRFAnalyzer handler or equivalent."""
        # I don't know how accurate this is.
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
    
    def info_and_dists(self,vecs=None, neg=True):
        """Returns the mutual information between spike arrival times and the projections along vecs."""
        if vecs is None:
            vecs = self.vvecs
        self.decide_bins(vecs=vecs) 
        
        pv = np.zeros(self.nbins*np.ones(self.ndim)) # prob dist of projections
        pvt = np.zeros_like(pv) # prob dist of projections given spike (t for trigger)
        for stim, sp in self.handler.generator():
            proj = vecs.dot(stim)
            projbin = tuple([self.bin_ind(proj[d],d) for d in range(self.ndim)]) # tuple indexes one entry; array does something else
            pv[projbin] = pv[projbin] + 1
            pvt[projbin] = pvt[projbin] + sp
        
        nstims = np.sum(pv)
        nspikes = np.sum(pvt)
        pv = pv/nstims
        pvt = pvt/nspikes
        safepv = np.copy(pv)
        safepv[safepv==0] = 1./nstims # prevents divide by zero errors when 0/0 below
        info = 0 
        flatpv = safepv.flatten()
        flatpvt = pvt.flatten()
        for ii in range(len(flatpv)):
            info += (0 if flatpvt[ii] == 0 else flatpvt[ii]*np.log2(flatpvt[ii]/flatpv[ii]))
        factor = -1 if neg else 1
        return factor*info, pv, pvt
    
    def info(self, vecs=None, neg=True):
        return self.info_and_dists(vecs,neg)[0]
        
    
    def info_grad(self, vecs, neg=True):
        """Return the information as in infov, and the gradient of the same with respect to v.
        If neg, returns minus these things."""
        self.decide_bins(vecs=vecs)        
        
        pv = np.zeros(self.nbins*np.ones(self.ndim)) # prob dist of projections
        pvt = np.zeros_like(pv) # prob dist of projections given spike (t for trigger)
        # the averages have shape (nbins)**ndim * stimlength; i.e., one average per n-dimensional bin
        sv = np.zeros(tuple(np.concatenate([self.nbins*np.ones(self.ndim),[np.prod(self.handler.stimshape)]]))) # mean stim given projection
        svt = np.zeros_like(sv) # mean stim given projection and spike
        nstims = 0
        nspikes = 0
        for stim, sp in self.handler.generator():
            proj = vecs.dot(stim)
            projbin = tuple([self.bin_ind(proj[d],d) for d in range(self.ndim)]) # tuple indexes one entry; array does something else
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
        deriv = np.gradient(pvt/safepv) # uses 2nd order method
        deriv = np.stack(deriv) # deriv returns a list of arrays, convert to array
        # above is dthing/dbin_i, we want dthing/dx_i. The distinction may matter if bin widths vary between dimensions, otherwise it's just a scaling
        deriv = deriv/(self.binbounds[:,1]-self.binbounds[:,0])
                
        grad = np.sum(pv[np.newaxis,...,np.newaxis]*(svt-sv)[np.newaxis,...]*deriv[...,np.newaxis],axis=tuple(np.arange(self.ndim)))

        info = 0
        flatpv = safepv.flatten()
        flatpvt = pvt.flatten()
        for ii in range(len(flatpv)):
            info += (0 if flatpvt[ii] == 0 else flatpvt[ii]*np.log2(flatpvt[ii]/flatpv[ii]))
        factor = -1 if neg else 1
        return factor*info, factor*grad
    
    def grad_ascent(self, vecs, rate, gtol=1e-6, maxiter=100):
        """Runs simple gradient ascent with a constant rate, until either the tolerance is achieved or maxiter iterations."""
        gnorm = 2*gtol
        it=0
        infohist=[]
        print('Info             Gradient norm')
        for it in range(maxiter):
            info, grad = self.info_grad(vecs,neg=False)
            infohist.append(info)
            gnorm = np.linalg.norm(grad)
            if gnorm<gtol:
                break
            print(str(info)+'  '+str(gnorm))
            vecs = vecs + rate*grad 
        print(str(info)+'  '+str(gnorm))
        if gnorm<gtol:
            mes = "Converged to desired precision."
        else:
            mes = "Did not converge to desired precision."
        return SimpleResult(vecs, mes, history=infohist)
        
        
    def line_max_backtrack(self, vecs, initinfo, grad, params=None):
        if params is None:
            params = BacktrackingParams()
        bestinfo=initinfo
        step = params.maxstep
        beststep = 0
        goodenough = np.linalg.norm(grad)*params.acceptable
        for it in range(params.maxiter):
            newinfo = self.info(vecs+step*grad, neg=False)
            if newinfo-initinfo > goodenough*step:
                print("Satisficed with step size " + str(step), " on iteration " + str(it))
                return step
            if newinfo > bestinfo:
                bestinfo = newinfo
                beststep = step
                print("Found new best step size " + str(beststep) + " with info " + str(bestinfo))
            step = step*params.reducefactor
        print("Updating with best found step size " + str(beststep))
        return beststep
    
    def GA_with_linemax(self, vecs, gtol=1e-5, maxiter=100, params=None):
        gnorm = 2*gtol
        it=0
        infohist=[]
        print('Info             Gradient norm')
        for it in range(maxiter):
            info, grad = self.info_grad(vecs,neg=False)
            assert infohist==[] or info > infohist[-1]
            infohist.append(info)
            gnorm = np.linalg.norm(grad)
            if gnorm < gtol:
                break
            print(str(info)+'  '+str(gnorm))
            step = self.line_max_backtrack(vecs, info, grad, params)
            if step == 0:
                print("No improvement found in direction of gradient.")
                break
            vecs = vecs + step*grad
        print(str(info)+'  '+str(gnorm))
        if gnorm<gtol:
            mes = "Converged to desired precision."
        else:
            mes = "Did not converge to desired precision."
        return SimpleResult(vecs, mes, history=infohist)
        
    def optimize(self, method='BFGS', rate=1e-6, maxiter=100):
        if method == 'BFGS':
            result = minimize(self.info_grad,self.vecs,method=method, jac=True, options={'disp':True, 'maxiter':maxiter})
        elif method == 'Nelder-Mead':
            result = minimize(self.info, self.vecs, method=method, options={'disp':True})
        elif method == 'GA':
            result = self.grad_ascent(self.vecs,rate, maxiter=maxiter)
        elif method == 'GA_with_linemax':
            result = self.GA_with_linemax(self.vecs, maxiter=maxiter)
        else:
            return SimpleResult(self.vecs, "No valid method provided. Did nothing.")
        
        print(result.message)
        self.vecs = result.x
        return result
    
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
        