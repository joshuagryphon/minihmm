#!/usr/bin/env python
"""Utilities for changing representations of observation sequences and/or models.
The primary functionality of the tools in this module is to translate Nth-order
HMMs into equivalent first order HMMs.


Notes on model reduction
------------------------
In a high-order model, the ends of sequences introduce inhomogeneities, similar
to state priors in first-order models. For example, in a 3rd order model, the
first two states must be handled separately from those remaining:
 
.. math::

   P = p(S_1) * p(S_2 | S_1) * p(S_3 | S_2, S_1) * p(S_4 | S_3, S_2) *  ... * p(S_N |, S_{N-2}, S_{N-1})          
   
Rather than adding extra probability tables to represent these inhomogeneities,
:mod:`miniHMM` creates dummy states. For a third-order model, we would need
to add two dummy start states in high-order space, $ S_{dummy1} $ and $ S_{dummy2} $.

We rephrase the model above as:

.. math::

   P = p(S_1 | S_{dummy2}, S_{dummy1}) * p(S_2 | S_1, S_{dummy2}, ) * p(S_3 | S_2, S_1) * p(S_4 | S_3, S_2) *  ... * p(S_N |, S_{N-2}, S_{N-1})

Then:

 - The prior probabilities of every state pair are set to zero, except
   $ p(S_{dummy2}, S_{dummy1}) $, which is  set to one.
   
 - The transition probabilities of p(S_i | S_{dummy2}, S_{dummy1}) are set to
   the prior probabilities of P(S_i) in the previous model description
   
 - Transition probabilities of p(S_i | S_{i-1}, S_{dummy2}) are set to the 
   observed values for P(S_i | S_{i-1}) in the previous model description

:mod:`miniHMM` contains tools for creating these dummy states, and remapping 
state sequences accordingly.



Workflow
--------

A number of different workflows are (will be) possible depending upon what type of
starting information you have:


If you have known series of states
..................................

In this case, it is easiest to transcode the state sequences into first-order
space, and then calculate the model parameters from the known state sequences.
To do this, call :func:`lower_stateseq_orders`, to re-map state sequences
into the reduced model space, then calculate transition probabilities as
for a first order model.


If high-order model parameters are known, but state sequences are not
.....................................................................

We don't support this yet, but will have some auto-mapping tools to move
between high-order and first-order models


If series of states are unknown, and training is required
.........................................................

We don't support this yet, either. Recommended regimen will be to train a
naive first-order model, then use those parameters to initialize a second-order
model, refine via re-training, and repeat until Nth order is reached.

This will require lots of translation of model parameters between spaces.


                          
.. autosummary::
       
   get_state_mapping
   lower_stateseq_orders
   raise_stateseq_orders
"""
import warnings
import itertools
import numpy



class ModelReducer(object):
    """Utility class for reducing high-order HMMs to equivalent first-order HMMs.


    Attributes
    ----------
    starting_order : int
        Order of starting model

    high_order_states : int
        Number of states, in high-order space

    high_states_to_low : dict
        Dictionary mapping high-order states to tuples of low-order states

    low_states_to_high : dict
        Dicitonary mapping tuples of low-order states to high-order states
    """

    def __init__(self, starting_order, num_states):

        if num_states < 1:
            raise ValueError("Must have >= 1 state in model.")
        if starting_order < 1:
            raise ValueError("Cannot reduce order of 0-order model")
        elif starting_order == 1:
            warnings.warn("Reducing a first-order model doesn't make much sense.", UserWarning)
 
        self.starting_order    = starting_order
        self.high_order_states = num_states
        self._dummy_states     = self._get_dummy_states()
        self.high_states_to_low, self.low_states_to_high = self._get_state_mapping()

    @staticmethod
    def transcode_sequences(sequences, alphadict):
        """Transcode a sequence from one alphabet to another
        
        Parameters
        ----------
        sequences : iterable
            Iterable of sequences (each, itself, an iterable like a list et c)
            
        alphadict : dict-like
            Dictionary mapping symbols input sequence to symbols in output sequence
            
        Returns
        -------
        list of :class:`numpy.ndarray`
            List of each transcoded sequence
        """
        return [numpy.array([alphadict[X] for X in Y]) for Y in sequences]

    def _get_dummy_states(self):
        """Create sorted lists of dummy states required to reduce `self` to an equivalent first-order model.
        
        By convention, dummy states are given negative indices in high-order space.
        Don't rely on this- it may change in the future
        
        Parameters
        ----------
        starting_order : int
            Starting order of HMM/MM
            
        Returns
        -------
        list
            List of new dummy states, sorted
        """
        return list(range(1 - self.starting_order, 0))
        
    def _get_state_mapping(self):
        """Create dicts mapping states between a high-order and a first-order HMM.

        Returns
        -------
        :class:`dict`
            Forward state map, mapping high-order states to tuples of equivalent low-order states
            
        :class:`dict`
            Reverse state map, mapping new first-order states to tuples of high-order states
        """
        forward = {}
        reverse = {}
        states = list(range(self.high_order_states))
            
        c = 0
        
        # create maps for newly added start and end states
        for idx in range(self.starting_order - 1):
            root = self._dummy_states[idx:]
            for symbol in itertools.product(*([states]*(idx + 1))):
                tup = tuple(root + list(symbol))
                forward[tup] = c
                reverse[c]   = tup
                c += 1
        
        # combinations of true states
        for n, symbol in enumerate(itertools.product(*([states]*self.starting_order))):
            forward[symbol] = n + c
            reverse[c + n] = symbol
        
        return forward, reverse

    def _get_stateseq_tuples(self, state_seqs):
        """Remap a high-order sequence of states into tuples for use in a low-order model, adding dummy start states
        
        Notes
        -----
        This does *not* remap those tuples into lower state spaces. Use
        :func:`lower_stateseq_orders`, which wraps this function, for that
        
        
        Parameters
        ----------
        state_seqs : list of list-like
            List of state sequences

        Returns
        -------
        list of lists
            List of tuples of state sequences
        """
        dummy_states   = self._dummy_states
        starting_order = self.starting_order
        outseqs = []
        for n, inseq in enumerate(state_seqs):
            if (numpy.array(inseq) < 0).any():
                raise ValueError("Found negative state label in input sequence %s!" % n)
            
            baseseq = dummy_states + list(inseq)
            outseqs.append([tuple(baseseq[idx:idx+starting_order]) for idx in range(0, len(baseseq) - starting_order + 1)])
        
        return outseqs 
        
    def lower_stateseq_orders(self, state_seqs):
        """Map a high-order sequence of states into an equivalent first-order state sequence,
        creating dummy states as necessary and dictionaries that map states between
        high and first-order spaces.
        
        
        Parameters
        ----------
        state_seqs : list of list-like
            List of state sequences, each given in high-order space
            
        Returns
        -------
        list of :class`numpy.ndarray`
            List of state sequences, represented in first-order space

           
        See also
        --------
        raise_stateseq_orders
        """
        dummy_states = self._dummy_states
        state_map    = self.high_states_to_low
            
        tuple_seqs = self._get_stateseq_tuples(state_seqs)
        return ModelReducer.transcode_sequences(tuple_seqs, state_map)
            
    def raise_stateseq_orders(self, state_seqs):
        """Map a state sequence from first-order space back to original high-order space
        
        Parameters
        ----------    
        state_seqs : list
            List of high-order state sequences

        Returns
        -------
        list of :class:`numpy.ndarray`
            State sequences, in high-order space
            
            
        See also
        --------
        lower_stateseq_orders
        """
        ltmp = []
        for t in transcode_sequences(state_seqs, self.low_states_to_high):
            ltmp.append([X[-1] for X in t])
        
        return ltmp
        
    # def raise_parameter_order()
    #     pass


    def remap_emission_factors(self, emission_probs):
        """Map emission probabilities from high-order space to equivalent reduced space

        Parameters
        ----------
        emission_probs : list of :class:`~minihmm.factors.AbstractFactor`
            List of emission probabilities, indexed by state in high-order space.

        Returns
        -------
        list
            list of Factors, indexed by states in equivalent first-order
        """
        reverse_state_map = self.low_states_to_high

        # make empty list length of new states
        ltmp = [None] * len(reverse_state_map)

        # populate list
        for newstate, state_tuple in reverse_state_map.items():
            ltmp[newstate] = emission_probs[state_tuple[-1]]

        return ltmp

    def get_pseudocount_array(self):
        """Return a valid pseudocount array for transition tables in first-order space,
        where *valid* stipulates that cells corresponding transitions that
        cannot exist in high-order space are set to zero.

        Returns
        -------
        :class:`numpy.ndarray`
            Square matrix
        """
        raise NotImplementedError()

 

#TODO :  implement and test
#def lower_parameter_order(states,
#                          starting_order   = 2,
#                          state_map        = None,
#                          state_priors     = None,
#                          transition_probs = None,
#                          emission_probs   = None,
#                         ):
#    """Create states and probability tables that map a high-order HMM to an equivalent first-order HMM
#     
#     
#    Parameters
#    ----------
#    states : int
#        Number of model states in high-order space
#     
#    starting_order : int, optional
#        Order of starting HMM/MM (Default: 2)
#         
#    state_priors : class:`numpy.ndarray`, optional
#        Probabilities of starting in any given state
#     
#    transition_probs : :class:`numpy.ndarray`, optional
#        Probabilities of transitions between states
#     
#     
#    Returns
#    -------
#    dict
#        Dictionary of objects describing reduced model:
#         
#            =================  ========================================  =================================================
#            Key                Type                                      Contains
#            -----------------  ----------------------------------------  -------------------------------------------------
#            statemap_forward   :class:`dict`                             Maps tuples of high-order states to new 1st-order
#                                                                         states
# 
#            statemap_reverse   :class:`dict`                             Maps new 1st-order states to corresponding
#                                                                         high-order states
#                                                                        
#            state_priors       :class:`numpy.ndarray`                    Mapping of 2nd order state priors to 1st order
#                                                                         space, with impossible starting points set to
#                                                                         zero
#                                                                        
#            transitions        :class:`numpy.ndarray`                    Mapping of 2nd order transition probabilities
#                                                                         to 1st order space, with impossible transitions
#                                                                         set to zero
#                                                                        
#            emissions          :class:`numpy.ndarray`                    Mapping of 2nd order emisison probabilities to
#                                                                         1st order space
#            =================  ========================================  =================================================
#    """
#    num_starting_states = len(states) 
#    starting_states     = range(num_starting_states)
#     
#    if state_priors is None:
#        state_priors = numpy.full(len(states), 1.0/num_starting_states)
#         
#    if transition_probs is None:
#        transition_probs = numpy.full(([num_starting_states]*(1 + starting_order)),
#                                      1.0/num_starting_states)
# 
#    dtmp = {}    
#    fullstates = copy.deepcopy(states)
#     
#    for n in range(starting_order-1):
#        fullstates.append("start%s" % n)
#        fullstates.append("end%s" % n)
# 
#    statemap_forward, statemap_reverse = map_states(fullstates)
#    dtmp["statemap_forward"] = statemap_forward
#    dtmp["statemap_reverse"] = statemap_reverse
#     
#    num_new_states = len(fullstates)
#     
#    new_state_priors = numpy.zeros(num_new_states, dtype=float)
#    new_transprobs   = numpy.zeros((num_new_states, num_new_states),
#                                   dtype = float)
#     
#    # transition probabilities
#    statepaths = itertools.product( *([starting_states]* (starting_order + 1)))
#    # remap transition probabilities to 1st order space
#    for path in statepaths:
#        old_startstate = path[:-1]
#        old_endstate   = path[1:]
#        new_transprobs[statemap_forward[old_startstate],
#                       statemap_forward[old_endstate]] = transition_probs[path]
#     
#    # add probabilities for added start, end states
#    for i in range(starting_order-1):
#        base = ["start%s" % X for X in range(i)]
#        for remaining in itertools.product( *([starting_states]* starting_order)):
#            idx = tuple(base + list(remaining))
#            fromstate = idx[:-1]
#            tostate   = idx[1:]
#             
#            # MARGINALIZE APPROPRIATELY
#            new_transprobs[statemap_forward[fromstate],
#                           statemap_forward[tostate]] = None 
#     
#    # remap state priors
#    for i in range(num_starting_states):
#        baseidx = ["start%s" % X for X in range(starting_order)]
#        idx = statemap_forward[tuple(baseidx + [i])]
#        new_state_priors[statemap_forward[idx]] = state_priors[i]
#         
#    # remap emissions - we stipulate that high order HMMs be first-order in emissions,
#    # so this is a fairly direct remapping
#    if emission_probs is not None:        
#        new_emission_probs = [None]*num_new_states
#        for i in range(num_new_states):
#            new_emission_probs[i] = emission_probs[statemap_reverse[i][:-1]]
#
#        dtmp["emissions"] = new_emission_probs
#     
#    dtmp["state_priors"] = new_state_priors
#    dtmp["transitions"]  = new_transprobs
#    return dtmp
