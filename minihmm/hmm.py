#!/usr/bin/env python
"""Contains classes that represent Hidden Markov Models.
These encapsulate methods for:

    #. Assigning state labels to sequences of observations, decoding via
       the Viterbi algorithm or by posterior decoding.
    
    #. Computing the probability of observing a sequence of observations via
       the forward algorithm
    
    #. Generating sequences of observations
    
All classes here support multivariate and univariate emissions (observation
sequences) that can be continuous or discrete. 

Training utilities for estimating model parameters may be found
in :py:mod:`minihmm.training`


Important classes
-----------------
|FirstOrderHMM|
    A First-Order HMM using ordinary precision math, suitable for most purposes

|HighPrecisionFirstOrderHMM|
    An arbitrary-precision HMM built upon the :py:obj:`mpmath` library.
    Far, far slower than |FirstOrderHMM|, but accurate in emergencies


References
----------
[Durbin1998]
    Durbin R et al. (1998). Biological sequence analysis: Probabilistic models
    of proteins and nucleic acids. Cambridge University Press, New York.
    ISBN 978-0-521-62971-3

[Rabiner1989]
    Rabiner, LR (1989). A Tutorial on Hidden Markov Models and Selected Applications
    in Speech Recognition. Proceedings of the IEEE, 77(2), pp 257-286

[WikipediaForwardBackward]
    http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

[WikipediaViterbi]
    http://en.wikipedia.org/wiki/Viterbi_algorithm
"""
import numpy
import copy
from minihmm.factors import AbstractGenerativeFactor, AbstractTableFactor

# # Enable or disable HighPrecisionFirstOrderHMM
# have_mpmath = True
# try:
#     import mpmath as mp
#     mpfdtype = numpy.dtype(mp.mpf)
#     to_mpf = numpy.vectorize(mp.mpf,otypes=(mpfdtype,),doc="Convert a numpy array to an array of arbitrary-precision floats")
#     mp_exp = numpy.vectorize(mp.exp,otypes=(mpfdtype,),doc="Exponentiate a numpy array with arbitrary precision")
#     mp_log = numpy.vectorize(mp.log,otypes=(mpfdtype,),doc="Take log of a numpy array with arbitrary precision")
# except ImportError:
#     have_mpmath = False
    

class FirstOrderHMM(AbstractGenerativeFactor):
    """First-order homogeneous Hidden Markov Model.
    
    Observations/emissions can be multivariate
    """
    def __init__(self,
                 state_priors   = None,
                 emission_probs = None,
                 trans_probs    = None):
        """Create a |FirstOrderHMM|.
        
        Parameters
        ----------
        state_priors : |ArrayFactor|
            Probabilities of starting in any state
                                 
        emission_probs  : list of Factors, or a |CompoundFactor|
            Probability distributions describing the probabilities of observing
            any emission in each state. If a list, the types of factors need
            not be identical (e.g. some could be Gaussian, others T-distributed,
            et c) 
        
        trans_probs : |MatrixFactor|
            |MatrixFactor| describing transition probabilities from each state
            (first index) to each other state (second index).
        """
        assert len(state_priors) == len(emission_probs)
        assert len(state_priors) == len(trans_probs)
        self.num_states = len(state_priors)
        self.state_priors   = state_priors
        self.emission_probs = emission_probs
        self.trans_probs    = trans_probs
        self._logt = numpy.log(trans_probs.data)
    
    def __str__(self):
        return repr(self)
   
    def __repr__(self):
        return "<%s parameters:[%s]>" % (self.__class__.__name__,
                                         self.serialize())
    
    def serialize(self):
        ltmp = ["state_priors",
                self.state_priors.serialize(),
                "transitions",
                self.trans_probs.serialize(),
                "emissions"]
        ltmp.extend([X.serialize() for X in self.emission_probs])
        return "\t".join(ltmp)

    def deserialize(self, param_str):
        sp = param_str.search("state_priors\t")
        t  = param_str.search("transitions\t")
        e  = param_str.search("emissions\t")
        new_state_priors = self.state_priors.deserialize(param_str[sp+len("state_priors\t"):t])
        new_trans_probs  = self.trans_probs.deserialize(param_str[t+len("transition\ts"):e])
        remaining = param_str[e+len("emissions\t"):]
        remaining_items = remaining.strip("\n").split("\t")
        new_emission_probs = []
        items_per_factor = len(remaining_items) // self.num_states
        assert len(remaining_items) % self.num_states == 0 # this won't be true for heterogenous factors
        for i in range(self.num_states):
            nf_params = remaining_items[i*items_per_factor:(i+1)*items_per_factor]
            new_factor = self.trans_probs.deserialize("\t".join(nf_params))
            new_emission_probs.append(new_factor)

        return self.__class__(new_state_priors,new_emission_probs,new_trans_probs)

    def probability(self, emission):
        """Compute the probability of observing a sequence of emissions.
        This number is likely to undeflow for long sequences.

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            Probability of sequence of emissions
        """
        return numpy.exp(self.logprob(emission))
    
    def logprob(self, emission):
        """Compute the log probability of observing a sequence of emissions.

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            log probability of sequence of emissions
        """
        return self.fast_forward(emission)

    def fast_forward(self, emissions):
        """Compute the log probability of observing a sequence of emissions.
        
        More memory efficient implementation of the forward algorithm, retaining
        only the probability of the terminal and penultimate states at each step.
        This implementation is not useful for posterior decoding, which requires
        the probability of all intermediate states. For that purpose, an 
        alternate implementation is provided by self.forward() 

        Numerical underflows are prevented by scaling probabilities at each step,
        following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            log probability of sequence of emissions
        """
        log_probability = 0
        T = self.trans_probs.data
        O0 = numpy.diag([self.emission_probs[X].probability(emissions[0]) for X in range(self.num_states)])
        prev_states_scaled = self.state_priors.data.dot(O0)

        for t in range(1,len(emissions)):
            Ot = numpy.diag([self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)])
            f  = prev_states_scaled.dot(T.dot(Ot))
            c  = f.sum()
            prev_states_scaled = f / c
            log_probability += numpy.log(c)
            
        return log_probability

    def forward(self, emissions):
        """Calculates the log-probability of observing a sequence of emissions,
        regardless of the state sequence, using the Forward Algorithm. This
        implementation also retains intermediate state information useful in
        posterior decoding or Baum-Welch training.
        
        Numerical underflows are prevented by scaling probabilities at each step,
        following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"
        
        Vectorized implementation from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        
        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            log probability of sequence of emissions
        
        numpy.ndarray
            [time x num_states] Array representing  scaled forward algorithm vector,
            indicating the forward probability of being in each state at time T,
            given the model and the observation trajectory from t = 0 to t = T
        
        numpy.ndarray
            [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
            The sum of the log of these equals the log probability of the observation sequence.
        """
        total_logprob, scaled_forward, _, scale_factors, _ = self.forward_backward(emissions,calc_backward=False)
        return total_logprob, scaled_forward, scale_factors
    
    def forward_backward(self, emissions, calc_backward=True):
        """Calculates the forward algorithm, the backward algorithm, and sufficient
        statistics useful in Baum-Welch calculations, all in factored and 
        vectorized forms. 
        
        Numerical underflows are prevented by scaling forward  probabilities at each
        step, following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"
        
        Vectorized implementation adapted from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        but using scaling as described in Rabiner1989 and Durbin1998, rather
        than the Wikipedia article
        
        
        Parameters
        ----------
        emissions : numpy.ndarray
            Observations
        
        calc_backward : bool, optional
            If True, perform backward pass and calculate Rabiner's ksi table
        
        
        Returns
        -------
        float
            log probability of sequence of emissions
        
        numpy.array
            Scaled forward algorithm vector of dim [time x num_states],
            indicating the forward probability of being in each state at time T,
            given the model and the observation trajectory from t = 0 to t = T

        numpy.ndarray
            Scaled backward algorithm vector of dim [time x num_states],
            indicating the backward probability of being in each state at time T,
            given the model and the observation trajectory from t = T to t = end

        numpy.ndarray
            [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
            The sum of the log of these equals the log probability of the observation sequence.
            
        numpy.ndarray
            [time x num_states x num_states] ksi table, as described in Rabiner 1989.
            At each time t, ksi[t,i,j] gives the posterior
            probability of transitioning from state i to state j. From this
            table it is trivial to derive the expected number of transitions 
            from state i to state j, or the posterior probability of being in
            state i or j at a given timepoint, by taking the appropriate sum.

        Notes
        -----
        This implementation casts everything to ``numpy.float128``. Whether this will
        actually force use of IEEE float128 depends on local C library implementations
        """
        # probability sequence indexed by timeslice. columns are end states
        scaled_forward = numpy.tile(numpy.nan,(len(emissions),self.num_states))
        scale_factors  = numpy.ones(len(emissions))
        T = numpy.array(self.trans_probs.data)
        O = []
    
        # initialize as prior + likelihood of emissions
        O.append([self.emission_probs[X].probability(emissions[0]) for X in range(self.num_states)])
        scaled_forward[0,:] = self.state_priors.data.dot(numpy.diag(O[0]))
        
        # can get underflows here from very improbable emissions
        #
        # can get nan for probability if f.sum() is 0, in other words, if a 
        # given observation is very improbable for all models and underflows 
        # for all models, then c = 0, and f/c = [nan,nan,...,nan]
        #
        # This then forces all future probabilities to be set to nan,
        # which messes up forward and backward calculations.
        # In this case, using HighPrecisionFirstOrderHMM will work,
        # but at a cost for speed
        for t in range(1,len(emissions)):
            O.append([self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)])
            f = scaled_forward[t-1,:].dot(T.dot(numpy.diag(O[t])))
            c = f.sum()
            scaled_forward[t,:] = f / c
            scale_factors[t] = c
        
        if calc_backward is True:
            # backward calc    
            scaled_backward = numpy.zeros((len(emissions),self.num_states))
            scaled_backward[-1,:] = 1.0 / scale_factors[-1] # <---- Wikipedia says not to scale final timestep; Rabiner & Durbin say to
            for t in range(len(emissions)-1)[::-1]:
                scaled_backward[t,:] = T.dot(numpy.diag(O[t+1]).dot(scaled_backward[t+1,:])) / scale_factors[t]

            # ksi calc 
            # NOTE: this is a complete calculation despite the fact that we
            #       are working in a scaled space, because the scale factors
            #       end up equaling 1/P(O|lambda), which means we can compute
            #       in scaled space and get an unscaled result if we do not
            #       divide by P(O|lambda)
            O = numpy.array(O)
            ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*T[None,:,:]*O[1:,None,:]

        else:
            scaled_backward = None
            ksi = None

        if numpy.isnan(scale_factors).any():
            total_logprob = -numpy.Inf
        else:  
            total_logprob = numpy.log(scale_factors).sum()
    
        return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi    
    
    def posterior_decode(self, emissions):
        """Find the most probable state for each individual state in the sequence
        of emissions, using posterior decoding. Note, this objective is distinct
        from finding the most probable sequence of states for all emissions, as
        is given in Viterbi decoding. This alternative may be more appropriate
        when multiple paths have similar probabilities, making the most likely
        path dubious.
        
        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        
        Returns
        -------
        numpy.ndarray
            An array of dimension [t x 1] of the most likely states at each point t
        
        numpy.ndarray
            An array of dimension [t x k] of the posterior probability of being
            in state k at time t 
        """
        _, forward, backward, scale_factors, _  = self.forward_backward(emissions)
        posterior_probs    = forward*backward*scale_factors[:,None]
        most_likely_states = (posterior_probs).argmax(1)
        
        return most_likely_states, posterior_probs
    
    def generate(self, length):
        """Generates a random sequence of states and emissions from the HMM
        
        Parameters
        ----------
        length : int
            Length of sequence to generate
        
        
        Returns
        -------
        numpy.ndarray
            Array of dimension [t x 1] indicating the HMM state at each timestep
        
        numpy.ndarray
            Array of dimension [t x Q] indicating the observation at each timestep.
            Q = 1 for univariate HMMs, or more than 1 if observations are multivariate.
            
        float
            Joint log probability of generated state and observation sequence.
            **Note**: this is different from the log probability of the observation
            sequence alone, which would be the sum of its joint probabilities
            with all possible state sequences.
        
        
        Notes
        -----
        The HMM can only generate sequences if all of its EmissionFactors
        are generative. I.e. if using |FunctionFactor| or |LogFunctionFactor| s,
        generator functions must be specified at their instantiation. See the
        documentation for |FunctionFactor| and |LogFunctionFactor| for help.
        """
        states    = []
        emissions = []
        
        states.append(self.state_priors.generate())
        emissions.append(self.emission_probs[states[0]].generate())
        
        logprob  = self.state_priors.logprob(states[0]) 
        logprob += self.emission_probs[states[0]].logprob(emissions[0])
        
        
        for i in range(1,length):
            new_state = self.trans_probs.generate(states[i-1])
            new_obs   = self.emission_probs[new_state].generate()
            
            states.append(new_state)
            emissions.append(new_obs)
            
            logprob += self.trans_probs.logprob(states[-1],new_state)
            logprob += self.emission_probs[new_state].logprob(new_obs)
        
        return numpy.array(states).astype(int), numpy.array(emissions), logprob

    def sample(self, emissions, num_samples=1):
        """Sample state sequences from the distribution P(states | emissions),
        by tracing backward through the matrix of forward probabilities. 
        See Durbin1997 ch 4.3, section "Probabilistic sampling of aligments"

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        num_samples : int, optional
            Number of state paths to generate (Default: 1)


        Returns
        -------
        list
            List of state paths
        """
        L = len(emissions)
        T = self.trans_probs.data

        total_logprob, scaled_forward, _, scale_factors, _ = self.forward_backward(emissions, calc_backward=False)
        randos = numpy.random.random(size=(num_samples,L))

        paths = []
        for n in range(num_samples):
            my_path = numpy.full(len(emissions), -1)

            # because probabilty at all steps is scaled to one, we can just 
            # examine cumsum of final step to start
            last_state = (scaled_forward[-1,:].cumsum() >= randos[n,0]).argmax()
            my_path[-1] = last_state
            
            for i in range(1, L):
                pvec = T[:,last_state] \
                       * hmm.emission_probs[last_state].probability(emissions[-i]) \
                       * scaled_forward[-i-1,:] \
                       / scaled_forward[-i,last_state] \
                       / scale_factors[-i]

                last_state = (pvec.cumsum() >= randos[n, i]).argmax()
                my_path[-i-1] = last_state

            paths.append(my_path)

        return paths

#     commented out - this actually samples according to joint distribution P(states,obs), NOT P(states|obs)
#     # TODO: test
#     def sample(self, emissions):
#         """Probabilistically sample state sequences from the joint distribution
#         P(emissions, states), using a modified Viterbi algorithm
#         
#         See Durbin1997 ch 4.3
# 
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Sequence of observations
#         
#         Returns
#         -------
#         dict
#             Results of decoding, with the following keys:
#             
#             `states`
#                 :class:`numpy.ndarray`. Decoded labels for each position in
#                 emissions[start:end]
#         
#             `logprob`
#                 :class:`float`. Joint log probability of sequence of
#                 `emissions[start:end]` and the decoded state sequence
#         """
#         rvals = numpy.random.random(len(emissions))
#         T     = self._logt
# 
#         prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[0])\
#                                   for X in range(self.num_states)])
#         
#         newstate   = (prev_probs.cumsum() >= rvals[0]).argmax()
#         state_path = [newstate]
#         logprob    = prev_probs[newstate] 
#         
#         for r, x in zip(rvals, emissions)[1:]:
#             emission_logprobs = numpy.array([X.logprob(x) for X in self.emission_probs])
#             current_probs     = T[state_path[-1],:] + emission_logprobs + logprob
#             newstate          = (current_probs.cumsum() >= r).argmax()
#             
#             state_path.append(newstate)
#             logprob += current_probs[newstate]
#         
#         dtmp = {
#             "sampled_states"  : state_path,
#             "logprob"         : logprob,
#         }
#         return dtmp
    
    def viterbi(self, emissions, start=0, end=None):
        """Finds the most likely state sequence underlying a set of emissions
        using the Viterbi algorithm. Also returns the natural log probability
        of that state sequence.
        
        See http://en.wikipedia.org/wiki/Viterbi_algorithm
        
        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        start : int, optional
            Starting point in emissions to consider (Default: 0)

        end : int, optional
            Ending point in emissions to consider (Default: None, end of sequence)
        

        Returns
        -------
        dict
            Results of decoding, with the following keys:
            
            `viterbi_states`
                :class:`numpy.ndarray`. Decoded labels for each position in
                emissions[start:end]
        
            `logprob`
                :class:`float`. Joint log probability of sequence of
                `emissions[start:end]` and the decoded state sequence
        
            `state_paths`
                dict[state] = log probability of final symbol being in that state
        """
        state_dict = { K : [K] for K in range(self.num_states) }
        prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[start])\
                                  for X in range(self.num_states)])

        T = self._logt
        for x in emissions[start+1:end]:
            new_state_dict = {}
            emission_logprobs = numpy.array([X.logprob(x) for X in self.emission_probs])
            current_probs     = T + emission_logprobs[None,:] + prev_probs[:,None]
            
            new_probs = current_probs.max(0)
            new_state_dict = { X : state_dict[current_probs[:,X].argmax()] + [X] \
                                   for X in range(self.num_states) }

            prev_probs = new_probs
            state_dict = new_state_dict
        
        final_label   = prev_probs.argmax()
        total_logprob = prev_probs.max()
        states = numpy.array(state_dict[final_label])
        
        dtmp = {
            "viterbi_states"  : states,
            "state_paths"     : state_dict,
            "logprob"         : total_logprob,
        }
        return dtmp


# class HighOrderHMM(FirstOrderHMM):
#     """Higher-order homogeneous Hidden Markov Model.
#     """
#     def __init__(self,state_priors,emission_probs,trans_probs):
#         """Create a |HighOrderHMM|.
#         
#         Parameters
#         ----------
#         state_priors : list of Factors
#             Factors of increasing complexity to describe the probabilities
#             of the first 0..`n-1` symbols in the chain, where `n` is the
#             order of the HMM.
#                                  
#         emission_probs  : list of Factors, or a |CompoundFactor|
#             Probability distributions describing the probabilities of observing
#             any emission in each state. If a list, the types of factors need
#             not be identical (e.g. some could be Gaussian, others T-distributed,
#             et c) 
#         
#         trans_probs : |MatrixFactor|
#             `n`-dimensional |MatrixFactor| describing transition probabilities
#             between states, where `n` is the order of the HMM. Dimensions
#             should be ordered temporally, such that the final dimension
#             corresponds to the state transitioned to
#         """
#         self.order = self.trans_probs.ndim - 1
#         assert len(state_priors) == self.order
#         assert len(state_priors[0]) == len(emission_probs)
#         assert len(state_priors[0]) == len(trans_probs[0])
#         self.num_states = len(state_priors[0])
#         self.state_priors   = state_priors
#         self.emission_probs = emission_probs
#         self.trans_probs    = trans_probs
# 
#     @NotImplemented
#     def fast_forward(self,emissions):
#         """Compute the log probability of observing a sequence of emissions.
#          
#         More memory efficient implementation of the forward algorithm, retaining
#         only the probability of the terminal and penultimate states at each step.
#         This implementation is not useful for posterior decoding, which requires
#         the probability of all intermediate states. For that purpose, an 
#         alternate implementation is provided by self.forward() 
#  
#         Numerical underflows are prevented by scaling probabilities at each step,
#         following the procedure given in Rabiner (1989), "A Tutorial on Hidden
#         Markov Models and Selected Applications in Speech Recognition"
#  
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Sequence of observations
#          
#         Returns
#         -------
#         float
#             log probability of sequence of emissions
#         """
#         log_probability = 0
#         T = self.trans_probs.data
#         O0 = numpy.diag([self.emission_probs[X].probability(emissions[0]) for X in range(self.num_states)])
#         prev_states_scaled = self.state_priors.data.dot(O0)
#  
#         for t in range(1,len(emissions)):
#             Ot = numpy.diag([self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)])
#             f  = prev_states_scaled.dot(T.dot(Ot))
#             c  = f.sum()
#             prev_states_scaled = f / c
#             log_probability += numpy.log(c)
#              
#         return log_probability
# 
#     @NotImplemented    
#     def forward_backward(self,emissions,calc_backward=True):
#         """Calculates the forward algorithm, the backward algorithm, and sufficient
#         statistics useful in Baum-Welch calculations, all in factored and 
#         vectorized forms. 
#         
#         Numerical underflows are prevented by scaling forward  probabilities at each
#         step, following the procedure given in Rabiner (1989), "A Tutorial on Hidden
#         Markov Models and Selected Applications in Speech Recognition"
#         
#         Vectorized implementation adapted from Wikipedia (2014-02-20):
#         http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
#         but using scaling as described in Rabiner1989 and Durbin1998, rather
#         than the Wikipedia article
#         
#         
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Observations
#         
#         calc_backward : bool, optional
#             If True, perform backward pass and calculate Rabiner's ksi table
#         
#         
#         Returns
#         -------
#         float
#             log probability of sequence of emissions
#         
#         numpy.array
#             Scaled forward algorithm vector of dim [time x num_states],
#             indicating the forward probability of being in each state at time T,
#             given the model and the observation trajectory from t = 0 to t = T
# 
#         numpy.ndarray
#             Scaled backward algorithm vector of dim [time x num_states],
#             indicating the backward probability of being in each state at time T,
#             given the model and the observation trajectory from t = T to t = end
# 
#         numpy.ndarray
#             [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
#             The sum of the log of these equals the log probability of the observation sequence.
#             
#         numpy.ndarray
#             [time x num_states x num_states] ksi table, as described in Rabiner 1989.
#             At each time t, ksi[t,i,j] gives the posterior
#             probability of transitioning from state i to state j. From this
#             table it is trivial to derive the expected number of transitions 
#             from state i to state j, or the posterior probability of being in
#             state i or j at a given timepoint, by taking the appropriate sum.
# 
#         Notes
#         -----
#         This implementation casts everything to ``numpy.float128``. Whether this will
#         actually force use of IEEE float128 depends on local C library implementations
#         """
#         # probability sequence indexed by timeslice. columns are end states
#         scaled_forward = numpy.tile(numpy.nan,(len(emissions),self.num_states))
#         scale_factors  = numpy.ones(len(emissions))
#         O = []
#     
#         # initialize as prior + likelihood of emissions
#         for i in range(self.order-1):
#             O.append([self.emission_probs[X].probability(emissions[i]) for X in range(self.num_states)])
#             # scaled_forward[i,:] = self.state_priors[i].data.dot(numpy.diag(O[i]))
#             f = numpy.zeros(self.num_states)
#             slice_indices = []
#             # TODO: this is incorrect. need to look up formula
#             for j in range(self.num_states+1):
#                 slice_indices = [slice(None) for _ in range(j)]
#                 slice_indices[i] = slice(j)
#                 f += self.state_priors[i].data[slices].dot(numpy.diag(O[i]))
# 
#             scale_factor = f.sum()
#             scale_factors[i] = scale_factor
#             scale_forward[i,:] = f / f.sum()
# 
#         # can get underflows here from very improbable emissions
#         #
#         # can get nan for probability if f.sum() is 0, in other words, if a 
#         # given observation is very improbable for all models and underflows 
#         # for all models, then c = 0, and f/c = [nan,nan,...,nan]
#         #
#         # This then forces all future probabilities to be set to nan,
#         # which messes up forward and backward calculations.
#         for t in range(1,len(emissions)):
#             O.append([self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)])
#             f = T.dot(numpy.diag(O[t]))
#             # TODO: deal with inhomogeneities for when t-1-i < self.order by marginalizing over dimension
#             for i in range(self.order):
#                 f = scaled_forward[t-1-i,:].dot(f)
#             c = f.sum()
#             scaled_forward[t,:] = f / c
#             scale_factors[t] = c
#         
#         if calc_backward is True:
#             # backward calc    
#             scaled_backward = numpy.zeros((len(emissions),self.num_states))
#             scaled_backward[-1,:] = 1.0 / scale_factors[-1] # <---- Wikipedia says not to scale final timestep; Rabiner & Durbin say to
#             for t in range(len(emissions)-1)[::-1]:
#                 b = T.dot(numpy.diag(O[t+1]))
#                     for i in range(self.order):
#                         b = b.dot(scaled_backward[t+i+1,:])) / scale_factors[t]
#                 scaled_backward[t,:] = b
# 
#             # ksi calc 
#             # NOTE: this is a complete calculation despite the fact that we
#             #       are working in a scaled space, because the scale factors
#             #       end up equaling 1/P(O|lambda), which means we can compute
#             #       in scaled space and get an unscaled result if we do not
#             #       divide by P(O|lambda)
#             O = numpy.array(O)
#             ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*T[None,:,:]*O[1:,None,:]
# 
#         else:
#             scaled_backward = None
#             ksi = None
# 
#         if numpy.isnan(scale_factors).any():
#             total_logprob = -numpy.Inf
#         else:  
#             total_logprob = numpy.log(scale_factors).sum()
#     
#         return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi
#         
#     @NotImplemented    
#     def generate(self,length):
#         """Generates a random sequence of states and emissions from the HMM
#         
#         Parameters
#         ----------
#         length : int
#             Length of sequence to generate
#         
#         
#         Returns
#         -------
#         numpy.ndarray
#             Array of dimension [t x 1] indicating the HMM state at each timestep
#         
#         numpy.ndarray
#             Array of dimension [t x Q] indicating the observation at each timestep.
#             Q = 1 for univariate HMMs, or more than 1 if observations are multivariate.
#         
#         
#         Notes
#         -----
#         The HMM can only generate sequences if all of its EmissionFactors
#         are generative. I.e. if using |FunctionFactor| or |LogFunctionFactor| s,
#         generator functions must be specified at their instantiation. See the
#         documentation for |FunctionFactor| and |LogFunctionFactor| for help.
#         """
#         states    = []
#         emissions = []
#         
#         states.append(self.state_priors.generate())
#         emissions.append(self.emission_probs[states[0]].generate())
#         
#         for i in range(1,length):
#             new_state = self.trans_probs.generate(states[i-1])
#             states.append(new_state)
#             emissions.append(self.emission_probs[new_state].generate())
#         
#         return numpy.array(states).astype(int), numpy.array(emissions)
# 
#     @NotImplemented
#     def viterbi(self,emissions,start=0,end=None,verbose=False):
#         """Finds the most likely state sequence underlying a set of emissions
#         using the Viterbi algorithm. Also returns the natural log probability
#         of that state sequence.
#         
#         See http://en.wikipedia.org/wiki/Viterbi_algorithm
#         
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Sequence of observations
#         
#         start : int, optional
#             Starting point in emissions to consider (Default: 0)
# 
#         end : int, optional
#             Ending point in emissions to consider (Default: None, end of sequence)
#         
#         verbose : bool
#             If True, the dictionary of log-probabilities at 
#             the final state is returned in addition to the
#             total log probability (Default: False)
# 
# 
#         Returns
#         -------
#         numpy.ndarray
#             Decoded labels for each position in emissions[start:end]
#                             
#         float
#             joint log probability of sequence of emissions[start:end]
#             and the decoded state sequence
#         
#         dict
#             dict[state] = log probability of final symbol
#             being in that state
#         """
#         state_dict = { K : [K] for K in range(self.num_states) }
#         prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[0])\
#                                   for X in range(self.num_states)])
# 
#         T = numpy.log(self.trans_probs.data)
#         for x in emissions[start+1:end]:
#             new_state_dict = {}
#             emission_logprobs = numpy.array([X.logprob(x) for X in self.emission_probs])
#             current_probs     = T + emission_logprobs[None,:] + prev_probs[:,None]
#             
#             new_probs = current_probs.max(0)
#             new_state_dict = { X : state_dict[current_probs[:,X].argmax()] + [X] \
#                                    for X in range(self.num_states) }
# 
#             prev_probs = new_probs
#             state_dict = new_state_dict
#         
#         final_label   = prev_probs.argmax()
#         total_logprob = prev_probs.max()
#         states = numpy.array(state_dict[final_label])
#         if verbose is False:
#             return states, total_logprob
#         else:
#             return states, total_logprob, state_dict


class FirstOrderMM(AbstractGenerativeFactor):
    """Implements a first-order homogeneous Markov Model.
    """
    def __init__(self,state_priors,trans_probs):
        """Create a |FirstOrderHMM|.
        
        Parameters
        ----------
        state_priors : |ArrayFactor|
            Probabilities of starting in any state
       
        trans_probs : |MatrixFactor|
            |MatrixFactor| describing transition probabilities from each state
            (first index) to each other state (second index).
        """
        assert len(state_priors) == len(trans_probs)
        self.num_states = len(state_priors)
        self.state_priors   = state_priors
        self.trans_probs    = trans_probs
    
    def __str__(self):
        return repr(self)
   
    def __repr__(self):
        return "<%s parameters:[%s]>" % (self.__class__.__name__,
                                         self.serialize())
    
    def serialize(self):
        raise NotImplementedError()

    def deserialize(self,param_str):
        raise NotImplementedError()

    def probability(self,sequence):
        """Compute the probability of observing a sequence.
        This number is likely to undeflow for long sequences.

        Parameters
        ----------
        sequence : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            Probability of sequence of emissions
        """
        return numpy.exp(self.logprob(sequence))
    
    def logprob(self,emission):
        """Compute the log probability of observing a sequence of sequence.

        Parameters
        ----------
        sequence : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            log probability of sequence of sequence
        """
        return self.fast_forward(emission)

    def forward(self,sequence):
        """Calculates the log-probability of observing a sequence,
        regardless of the state sequence, using the Forward Algorithm. This
        implementation also retains intermediate state information useful in
        posterior decoding or Baum-Welch training.
        
        Numerical underflows are prevented by scaling probabilities at each step,
        following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"
        
        Vectorized implementation from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        
        Parameters
        ----------
        sequence : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            log probability of sequence of sequence
        
        numpy.ndarray
            [time x num_states] Array representing  scaled forward algorithm vector,
            indicating the forward probability of being in each state at time T,
            given the model and the observation trajectory from t = 0 to t = T
        
        numpy.ndarray
            [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
            The sum of the log of these equals the log probability of the observation sequence.
        """
        total_logprob, scaled_forward, _, scale_factors, _ = self.forward_backward(sequence,calc_backward=False)
        return total_logprob, scaled_forward, scale_factors
    
    # TODO: test
    def forward_backward(self,sequence,calc_backward=True):
        """Calculates modified forward and backward algorithms for partially-observed sequences generated by Markov Models,
        as well as sufficient statistics useful in Baum-Welch calculations, all in factored and 
        vectorized forms. 
        
        Numerical underflows are prevented by scaling forward  probabilities at each
        step, following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"
        
        Vectorized implementation adapted from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        but using scaling as described in Rabiner1989 and Durbin1998, rather
        than the Wikipedia article
        
        
        Parameters
        ----------
        sequence : numpy.ndarray
            Observation sequence, possibly a partially-observed sequence, with any unobserved state set to -1.
        
        calc_backward : bool, optional
            If True, perform backward pass and calculate Rabiner's ksi table
        
        
        Returns
        -------
        float
            log probability of sequence of sequence
        
        numpy.array
            Scaled forward algorithm vector of dim [time x num_states],
            indicating the forward probability of being in each state at time T,
            given the model and the observation trajectory from t = 0 to t = T

        numpy.ndarray
            Scaled backward algorithm vector of dim [time x num_states],
            indicating the backward probability of being in each state at time T,
            given the model and the observation trajectory from t = T to t = end

        numpy.ndarray
            [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
            The sum of the log of these equals the log probability of the observation sequence.
            
        numpy.ndarray
            [time x num_states x num_states] ksi table, as described in Rabiner 1989.
            At each time t, ksi[t,i,j] gives the posterior
            probability of transitioning from state i to state j. From this
            table it is trivial to derive the expected number of transitions 
            from state i to state j, or the posterior probability of being in
            state i or j at a given timepoint, by taking the appropriate sum.

        Notes
        -----
        This implementation casts everything to ``numpy.float128``. Whether this will
        actually force use of IEEE float128 depends on local C library implementations
        """
        # probability sequence indexed by timeslice. columns are end states
        scaled_forward = numpy.zeros((len(sequence),self.num_states))
        scale_factors  = numpy.ones(len(sequence))
        O = []
    
        # initialize as prior + likelihood of sequence
        scaled_forward[0,sequence[0]] = self.state_priors[sequence[0]]
        
        # can get underflows here from very improbable sequence
        #
        # can get nan for probability if f.sum() is 0, in other words, if a 
        # given observation is very improbable for all models and underflows 
        # for all models, then c = 0, and f/c = [nan,nan,...,nan]
        #
        # This then forces all future probabilities to be set to nan,
        # which messes up forward and backward calculations.
        # In this case, using HighPrecisionFirstOrderHMM will work,
        # but at a cost for speed
        for t in range(1,len(sequence)):
            if sequence[t] == -1:
                f = scaled_forward[t-1,:].dot(self.trans_probs)
                c = f.sum()
                scaled_forward[t,:] = f / c
                scale_factors[t] = f
            else:
                f = (scaled_forward[t-1,:] * self.trans_probs[sequence[t-1],sequence[t]]).sum()
                scaled_forward[t,sequence[t]] = 1.0
                scale_factors[t] = f
        
        if calc_backward is True:
            # backward calc    
            scaled_backward = numpy.zeros((len(sequence),self.num_states))
            scaled_backward[-1,:] = 1.0 / scale_factors[-1] # <---- Wikipedia says not to scale final timestep; Rabiner & Durbin say to
            for t in range(len(sequence)-1)[::-1]:
                if sequence[t] == -1:
                    scaled_backward[t,:] = self.trans_probs.dot(scaled_backward[t+1,:]) / scale_factors[t]
                else:
                    scaled_backward[t,sequence[t]] = (self.trans_probs[sequence[t],sequence[t+1]] * scaled_backward[t+1]).sum() / scale_factors[t]

            # scale factors cancel out, returning Rabiner's ksi statistic
            ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*self.trans_probs[None,:,:]

        else:
            scaled_backward = None
            ksi = None

        if numpy.isnan(scale_factors).any():
            total_logprob = -numpy.Inf
        else:  
            total_logprob = numpy.log(scale_factors).sum()
    
        return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi    
    
    def posterior_decode(self,partial_sequence):
        """Find the most probable a posteriori state for each unfilled observation in `partial_sequence`
        
         .. note::
            This objective is distinct
            from finding the most probable sequence of states for all sequence, as
            is given in Viterbi decoding. This alternative may be more appropriate
            when multiple paths have similar probabilities, making the most likely
            path dubious.

            Also note, the posterior probabilities at observed positions TODO
        
        Parameters
        ----------
        partial_sequence : numpy.ndarray
            Sequence of observations, with unobserved states set to -1
        
        Returns
        -------
        numpy.ndarray
            An array of dimension [t x 1] of the most likely states at each point t
        
        numpy.ndarray
            An array of dimension [t x k] of the posterior probability of being
            in state k at time t 
        """
        _, forward, backward, scale_factors, _  = self.forward_backward(partial_sequence)
        posterior_probs    = forward*backward*scale_factors
        most_likely_states = (posterior_probs).argmax(1)
        
        return most_likely_states, posterior_probs
    
    def generate(self,length):
        """Generates a random sequence of states and sequence from the HMM
        
        Parameters
        ----------
        length : int
            Length of sequence to generate
        
        
        Returns
        -------
        numpy.ndarray
            Array of dimension [t x 1] indicating the MM state at each timestep
        """
        sequence = []
        
        sequence.append(self.state_priors.generate())
        
        for i in range(1,length):
            new_state = self.trans_probs.generate(sequence[i-1])
            sequence.append(new_state)
        
        return numpy.array(sequence).astype(int)

    def sample(self,partial_sequence):
        """Probabilistically sample state sequences from the HMM using a modified
        Viterbi algorithm, given a set of observations. This may be used to test
        the reliability of a Viterbi decoding, or of subregions of the Viterbi
        decoding. 
        
        See Durbin1997 ch 4.3

        Parameters
        ----------
        sequence : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        numpy.ndarray
            A sequence of states, sampled according to its probability under
            the model
        
        float
            joint log probability of sequence of sequence
            and the returned state sequence
        """
        raise NotImplementedError("FirstOrderHMM.sample() is not yet implemented")
    
    def viterbi(self,sequence,verbose=False):
        """Finds the most likely state sequence underlying a set of sequence
        using the Viterbi algorithm. Also returns the natural log probability
        of that state sequence.
        
        See http://en.wikipedia.org/wiki/Viterbi_algorithm
        
        Parameters
        ----------
        sequence : numpy.ndarray
            Partial sequence of states, with unobserved states set to -1
       
        verbose : bool
            If True, the dictionary of log-probabilities at 
            the final state is returned in addition to the
            total log probability (Default: False)


        Returns
        ------
        numpy.ndarray
            Decoded labels for each position in sequence[start:end]
                            
        float
            joint log probability of sequence of sequence[start:end]
            and the decoded state sequence
        
        dict
            dict[state] = log probability of final symbol
            being in that state
        """
        T = numpy.log(self.trans_probs.data)

        state_dict = { K : [K] for K in range(self.num_states) }
        if sequence[0] == -1:
            prev_probs = copy.deepcopy(self.state_priors.data)
        else:
            prev_probs = numpy.zeros(self.num_states)
            prev_probs[sequence[0]] = 1.0

        for n,t in enumerate(sequence):
            if  sequence[t] == -1:
                new_state_dict = {}
                current_probs  = T + prev_probs[:,None]
                new_probs = current_probs.max(0)
            else:
                new_probs = numpy.zeros_like(new_probs)
                new_probs[sequence[t]] = 1.0

            new_state_dict = { X : state_dict[current_probs[:,X].argmax()] + [X] \
                                   for X in range(self.num_states) }

            prev_probs = new_probs
            state_dict = new_state_dict
        
        final_label   = prev_probs.argmax()
        total_logprob = prev_probs.max()
        states = numpy.array(state_dict[final_label])
        if verbose is False:
            return states, total_logprob
        else:
            return states, total_logprob, state_dict

# class HighPrecisionFirstOrderHMM(FirstOrderHMM):
#     """First order HMM implementation that uses arbitrary-precision calculations
#     in the forward & backward algorithms. This sacrifices speed to avoid numerical
#     underflows when the HMM encounters observations that are extremely unlikely
#     under all observation models. This implementation relies upon mpmath for 
#     arbitrary-precision calculations.
#     
#     Before using, remember to set precision in mpath! Otherwise, no gains are made::
#     
#         >> import mpmath as mp
#         >> mp.dps = 200 # some number of decimal places of accuracy. Default: 15 for float64
#         
#     """
#     def __init__(self,*args):
#         """Create a |HighPrecisionFirstOrderHMM|. An AssertionError 
#         will be raised if mpmath library is not installed
#         
#         Parameters
#         ----------
#         state_priors : |ArrayFactor|
#             Probabilities of starting in any state
#                                  
#         emission_probs  : list
#             List of EmissionFactors describing the probabilities of observing
#             any emission in each state. Each list item can be a different types
#             of EmissionFactor
#         
#         trans_probs : |MatrixFactor|
#             |MatrixFactor| describing transition probabilities from each state
#             (first index) to each other state (second index).
#         """
#         assert have_mpmath is True
#         FirstOrderHMM.__init__(self,*args)
# 
#     def fast_forward(self,emissions):
#         """Compute the log probability of observing a sequence of emissions.
#         
#         More memory efficient implementation of the forward algorithm, retaining
#         only the probability of the terminal and penultimate states at each step.
#         This implementation is not useful for posterior decoding, which requires
#         the probability of all intermediate states. For that purpose, an 
#         alternate implementation is provided by self.forward() 
# 
#         Numerical underflows are prevented by scaling probabilities at each step,
#         following the procedure given in Rabiner (1989), "A Tutorial on Hidden
#         Markov Models and Selected Applications in Speech Recognition"
# 
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Sequence of observations
#         
#         Returns
#         -------
#         float
#             log probability of sequence of emissions
#         """
#         log_probability = mp.mpf('0')
#         T = to_mpf(self.trans_probs.data)
#         O0 = numpy.diag(mp_exp([self.emission_probs[X].logprob(emissions[0]) for X in range(self.num_states)]))
#         prev_states_scaled = to_mpf(self.state_priors.data.dot(O0))
# 
#         for t in range(1,len(emissions)):
#             Ot = numpy.diag(mp_exp([self.emission_probs[X].logprob(emissions[t]) for X in range(self.num_states)]))
#             f  = prev_states_scaled.dot(T.dot(Ot))
#             c  = f.sum()
#             prev_states_scaled = f / c
#             log_probability += mp.log(c)
#             
#         return numpy.float128(log_probability)
# 
#     def forward_backward(self,emissions,calc_backward=True):
#         """Calculates the forward algorithm, the backward algorithm, and sufficient
#         statistics useful in Baum-Welch calculations, all in factored and 
#         vectorized forms. 
#         
#         Numerical underflows are prevented by scaling forward  probabilities at each
#         step, following the procedure given in Rabiner (1989), "A Tutorial on Hidden
#         Markov Models and Selected Applications in Speech Recognition"
#         
#         Vectorized implementation adapted from Wikipedia (2014-02-20):
#         http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
#         but using scaling as described in Rabiner1989 and Durbin1998, rather
#         than the Wikipedia article
#         
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Observations
#         
#         calc_backward : bool, optional
#             If True, perform backward pass and calculate Rabiner's ksi table
#         
#         
#         Returns
#         -------
#         float
#             log probability of sequence of emissions
#         
#         numpy.ndarray
#             Scaled forward algorithm vector of dim [time x num_states],
#             indicating the forward probability of being in each state at time T,
#             given the model and the observation trajectory from t = 0 to t = T
# 
#         numpy.ndarray
#             Scaled backward algorithm vector of dim [time x num_states],
#             indicating the backward probability of being in each state at time T,
#             given the model and the observation trajectory from t = T to t = end
# 
#         numpy.ndarray
#             [time x 1] Array of scaling constants used at each step as described in Rabiner 1989.
#             The sum of the log of these equals the log probability of the observation sequence.
#             
#         numpy.ndarray
#             [time x num_states x num_states] ksi table, as described in Rabiner 1989.
#             At each time t, ksi[t,i,j] gives the posterior
#             probability of transitioning from state i to state j. From this
#             table it is trivial to derive the expected number of transitions 
#             from state i to state j, or the posterior probability of being in
#             state i or j at a given timepoint, by taking the appropriate sum.
#         """
#         # probability sequence indexed by timeslice. columns are end states
#         scaled_forward = to_mpf(numpy.zeros((len(emissions),self.num_states)))
#         scale_factors  = to_mpf(numpy.ones(len(emissions)))
#         T = to_mpf(self.trans_probs.data)
#         O = []
#     
#         # initialize as prior + likelihood of emissions
#         O.append(mp_exp([self.emission_probs[X].logprob(emissions[0]) for X in range(self.num_states)]))
#         scaled_forward[0,:] = to_mpf(self.state_priors.data).dot(numpy.diag(O[0]))
#         
#         for t in range(1,len(emissions)):
#             O.append(mp_exp([self.emission_probs[X].logprob(emissions[t]) for X in range(self.num_states)]))
#             f = scaled_forward[t-1,:].dot(T.dot(numpy.diag(O[t])))
#             c = f.sum()
#             scaled_forward[t,:] = f / c
#             scale_factors[t] = c
#         
#         if calc_backward is True:
#             # backward calc    
#             scaled_backward = to_mpf(numpy.ones((len(emissions),self.num_states)))
#             scaled_backward[-1,:] /= scale_factors[t] # <---- Wikipedia says not to scale; Rabiner & Durbin say so
#             for t in range(len(emissions)-1)[::-1]:
#                 scaled_backward[t,:] = T.dot(numpy.diag(O[t+1]).dot(scaled_backward[t+1,:])) / scale_factors[t]
# 
#             O   = numpy.array(O)
#             ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*T[None,:,:]*O[1:,None,:]
#         
#         else:
#             scaled_backward = None
#             ksi = None
# 
#         total_logprob = numpy.float128(mp_log(scale_factors).sum())
#     
#         return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi    
