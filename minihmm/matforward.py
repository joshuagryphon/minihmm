#!?usr/bin/env python
import numpy
from minihmm.hmm import *

def matforward(hmm, emissions, emission_prob_matrix, calc_backward=False):
    scaled_forward = numpy.tile(numpy.nan,(len(emissions),hmm.num_states))
    scale_factors  = numpy.ones(len(emissions))
    T = hmm.trans_probs.data
    O = []

    O = numpy.vstack([emission_prob_matrix[:,X] for X in emissions])

    scaled_forward[0,:] = hmm.state_priors.data.dot(numpy.diag(O[0,:]))
    
    for t in range(1,len(emissions)):
        f = scaled_forward[t-1,:].dot(T.dot(numpy.diag(O[t,:])))
        c = f.sum()
        scaled_forward[t,:] = f / c
        scale_factors[t] = c
    
    if calc_backward is True:
        # backward calc    
        scaled_backward = numpy.zeros((len(emissions),hmm.num_states))
        scaled_backward[-1,:] = 1.0 / scale_factors[-1] # <---- Wikipedia says not to scale final timestep; Rabiner & Durbin say to
        for t in range(len(emissions)-1)[::-1]:
            scaled_backward[t,:] = T.dot(numpy.diag(O[t+1,:]).dot(scaled_backward[t+1,:])) / scale_factors[t]

        # ksi calc 
        # NOTE: this is a complete calculation despite the fact that we
        #       are working in a scaled space, because the scale factors
        #       end up equaling 1/P(O|lambda), which means we can compute
        #       in scaled space and get an unscaled result if we do not
        #       divide by P(O|lambda)
        ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*T[None,:,:]*O[1:,None,:]

    else:
        scaled_backward = None
        ksi = None

    if numpy.isnan(scale_factors).any():
        total_logprob = -numpy.Inf
    else:  
        total_logprob = numpy.log(scale_factors).sum()

    return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi    

