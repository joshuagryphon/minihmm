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
from minihmm.factors import AbstractGenerativeFactor, AbstractTableFactor

# Enable or disable HighPrecisionFirstOrderHMM
have_mpmath = True
try:
    import mpmath as mp
    mpfdtype = numpy.dtype(mp.mpf)
    to_mpf = numpy.vectorize(mp.mpf,otypes=(mpfdtype,),doc="Convert a numpy array to an array of arbitrary-precision floats")
    mp_exp = numpy.vectorize(mp.exp,otypes=(mpfdtype,),doc="Exponentiate a numpy array with arbitrary precision")
    mp_log = numpy.vectorize(mp.log,otypes=(mpfdtype,),doc="Take log of a numpy array with arbitrary precision")
except ImportError:
    have_mpmath = False
    

class FirstOrderHMM(AbstractGenerativeFactor):
    """Implements a first-order homogeneous Hidden Markov Model.
    Multiple symbols per emission are permitted. All probabilities must be
    supplied as Factors (see above).
    """
    def __init__(self,state_priors,emission_probs,trans_probs):
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

    def deserialize(self,param_str):
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
        for i in range(num_states):
            nf_params = remaining_items[i*items_per_factor:(i+1)*items_per_factor]
            new_factor = self.trans_probs.deserialize("\t".join(nf_params))
            new_emission_probs.append(new_factor)

        return self.__class__(new_state_priors,new_emission_probs,new_trans_probs)

    def probability(self,emission):
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
    
    def logprob(self,emission):
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

    def fast_forward(self,emissions):
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

    def forward(self,emissions):
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
    
    def forward_backward(self,emissions,calc_backward=True):
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
    
    def posterior_decode(self,emissions):
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
    
    def generate(self,length):
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
        
        for i in range(1,length):
            new_state = self.trans_probs.generate(states[i-1])
            states.append(new_state)
            emissions.append(self.emission_probs[new_state].generate())
        
        return numpy.array(states).astype(int), numpy.array(emissions)

    def sample(self,emissions):
        """Probabilistically sample state sequences from the HMM using a modified
        Viterbi algorithm, given a set of observations. This may be used to test
        the reliability of a Viterbi decoding, or of subregions of the Viterbi
        decoding. 
        
        See Durbin1997 ch 4.3

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations
        
        Returns
        -------
        numpy.ndarray
            A sequence of states, sampled according to its probability under
            the model
        
        float
            joint log probability of sequence of emissions
            and the returned state sequence
        """
        pass
    
    def viterbi(self,emissions,start=0,end=None,verbose=False):
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
        
        verbose : bool
            If True, the dictionary of log-probabilities at 
            the final state is returned in addition to the
            total log probability (Default: False)


        Returns
        -------
        numpy.ndarray
            Decoded labels for each position in emissions[start:end]
                            
        float
            joint log probability of sequence of emissions[start:end]
            and the decoded state sequence
        
        dict
            dict[state] = log probability of final symbol
            being in that state
        """
        state_dict = { K : [K] for K in range(self.num_states) }
        prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[0])\
                                  for X in range(self.num_states)])

        T = numpy.log(self.trans_probs.data)
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
        if verbose is False:
            return states, total_logprob
        else:
            return states, total_logprob, state_dict


class HighPrecisionFirstOrderHMM(FirstOrderHMM):
    """First order HMM implementation that uses arbitrary-precision calculations
    in the forward & backward algorithms. This sacrifices speed to avoid numerical
    underflows when the HMM encounters observations that are extremely unlikely
    under all observation models. This implementation relies upon mpmath for 
    arbitrary-precision calculations.
    
    Before using, remember to set precision in mpath! Otherwise, no gains are made::
    
        >> import mpmath as mp
        >> mp.dps = 200 # some number of decimal places of accuracy. Default: 15 for float64
        
    """
    def __init__(self,*args):
        """Create a |HighPrecisionFirstOrderHMM|. An AssertionError 
        will be raised if mpmath library is not installed
        
        Parameters
        ----------
        state_priors : |ArrayFactor|
            Probabilities of starting in any state
                                 
        emission_probs  : list
            List of EmissionFactors describing the probabilities of observing
            any emission in each state. Each list item can be a different types
            of EmissionFactor
        
        trans_probs : |MatrixFactor|
            |MatrixFactor| describing transition probabilities from each state
            (first index) to each other state (second index).
        """
        assert have_mpmath is True
        FirstOrderHMM.__init__(self,*args)

    def fast_forward(self,emissions):
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
        log_probability = mp.mpf('0')
        T = to_mpf(self.trans_probs.data)
        O0 = numpy.diag(mp_exp([self.emission_probs[X].logprob(emissions[0]) for X in range(self.num_states)]))
        prev_states_scaled = to_mpf(self.state_priors.data.dot(O0))

        for t in range(1,len(emissions)):
            Ot = numpy.diag(mp_exp([self.emission_probs[X].logprob(emissions[t]) for X in range(self.num_states)]))
            f  = prev_states_scaled.dot(T.dot(Ot))
            c  = f.sum()
            prev_states_scaled = f / c
            log_probability += mp.log(c)
            
        return numpy.float128(log_probability)

    def forward_backward(self,emissions,calc_backward=True):
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
        
        numpy.ndarray
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
        """
        # probability sequence indexed by timeslice. columns are end states
        scaled_forward = to_mpf(numpy.zeros((len(emissions),self.num_states)))
        scale_factors  = to_mpf(numpy.ones(len(emissions)))
        T = to_mpf(self.trans_probs.data)
        O = []
    
        # initialize as prior + likelihood of emissions
        O.append(mp_exp([self.emission_probs[X].logprob(emissions[0]) for X in range(self.num_states)]))
        scaled_forward[0,:] = to_mpf(self.state_priors.data).dot(numpy.diag(O[0]))
        
        for t in range(1,len(emissions)):
            O.append(mp_exp([self.emission_probs[X].logprob(emissions[t]) for X in range(self.num_states)]))
            f = scaled_forward[t-1,:].dot(T.dot(numpy.diag(O[t])))
            c = f.sum()
            scaled_forward[t,:] = f / c
            scale_factors[t] = c
        
        if calc_backward is True:
            # backward calc    
            scaled_backward = to_mpf(numpy.ones((len(emissions),self.num_states)))
            scaled_backward[-1,:] /= scale_factors[t] # <---- Wikipedia says not to scale; Rabiner & Durbin say so
            for t in range(len(emissions)-1)[::-1]:
                scaled_backward[t,:] = T.dot(numpy.diag(O[t+1]).dot(scaled_backward[t+1,:])) / scale_factors[t]

            O   = numpy.array(O)
            ksi = scaled_forward[:-1,:,None]*scaled_backward[1:,None,:]*T[None,:,:]*O[1:,None,:]
        
        else:
            scaled_backward = None
            ksi = None

        total_logprob = numpy.float128(mp_log(scale_factors).sum())
    
        return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi    
