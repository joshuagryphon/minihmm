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
To do this, call :func:`lower_stateseq_orders`.


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
import copy
import warnings
import itertools
import numpy



def _get_dummy_states(starting_order):
    """Create sorted lists of dummy states required to reduce a model with
    `num_states` and `starting_order` to an equivalent first-order model.
    
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
    return list(range(1 - starting_order, 0))
    
def get_state_mapping(num_states, dummy_states, starting_order=2):
    """Create dicts mapping states between a high-order and a first-order HMM,
    adding dummy states to handle inhomogeneities at the starts of sequences.
    
    In the representation space of the high-order model, new start states
    are given negative indices.
    
    Parameters
    ----------
    num_states : int
        Number of states in high-order HMM/MM

    dummy_states : list
        Sorted list of dummy states. See :func:`_get_dummy_states`
    
    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)

    Returns
    -------
    :class:`dict`
        Forward state map, mapping high-order states to tuples of equivalent low-order states
        
    :class:`dict`
        Reverse state map, mapping new first-order states to tuples of high-order states
    """
    if num_states < 1:
        raise ValueError("Must have >= 1 state in model.")
    
    if starting_order < 1:
        raise ValueError("Cannot reduce order of 0-order model")
    elif starting_order == 1:
        warnings.warn("Reducing a first-order model doesn't make much sense.",UserWarning)
    
    forward = {}
    reverse = {}
    states = list(range(num_states))
        
    c = 0
    
    # create maps for newly added start and end states
    for idx in range(starting_order - 1):
        root = dummy_states[idx:]
        for symbol in itertools.product(*([states]*(idx + 1))):
            tup = tuple(root + list(symbol))
            forward[tup] = c
            reverse[c]   = tup
            c += 1
    
    # combinations of true states
    for n, symbol in enumerate(itertools.product(*([states]*starting_order))):
        forward[symbol] = n + c
        reverse[c + n] = symbol
    
    return forward, reverse

def _get_stateseq_tuples(state_seqs,
                         dummy_states, 
                         starting_order = 2):
    """Remap a high-order sequence of states into tuples for use in a low-order model,
    adding dummy start states
    
    Notes
    -----
    This does *not* remap those tuples into lower state spaces. Use
    :func:`reduce_stateseq_orders`, which wraps this function, for that
    
    
    Parameters
    ----------
    states : list
        List of states sequences
    
    num_states : int
        Number of states in high-order HMM/MM
    
    dummy_states : list
        Sorted list of dummy states. See :func:`_get_dummy_states`

    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)
    """
    outseqs = []
    for n, inseq in enumerate(state_seqs):
        if (numpy.array(inseq) < 0).any():
            raise ValueError("Found negative state label in input sequence %s!" % n)
        
        baseseq = dummy_states + list(inseq)
        outseqs.append([tuple(baseseq[idx:idx+starting_order]) for idx in range(0, len(baseseq) - starting_order + 1)])
    
    return outseqs 
    
def lower_stateseq_orders(state_seqs,
                          num_states,
                          starting_order = 2,
                          dummy_states   = None,
                          state_map      = None):
    """Map a high-order sequence of states into an equivalent first-order state sequence,
    creating dummy states as necessary and dictionaries that map states between
    high and first-order spaces.
    
    
    Parameters
    ----------
    states : list
        List of state sequences
    
    num_states : int
        Number of states in high-order HMM/MM

    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)
    
    newstarts : list, optional
        Sorted list of dummy states. If `None`, will be calculated using
        :func:`_get_dummy_states`
        
    state_map : dict, optional
        Dictionary mapping tuples of high-order states to low-order states.
        If `None`, will be calculated using :func:`get_state_mapping`
        
        
    Returns
    -------
    dict
        Dictionary containing the following keys:
        
        ======================   ===============================================
        **Key**                  **Value**
        ----------------------   -----------------------------------------------
        `dummy_states`           List of dummy states. Calculated if not
                                 supplied
                                 
        `state_map`              Dictionary mapping high order states to
                                 first-order states. Calculated if not supplied
        
        `state_seqs`             List of remapped state sequences, each a
                                 :class:`numpy.ndarray`
        ======================   ===============================================
        
        
    See also
    --------
    raise_stateseq_orders
    """
    if dummy_states is None:
        dummy_states = _get_dummy_states(starting_order)
        
    if state_map is None:
        state_map, _ = get_state_mapping(num_states,
                                         dummy_states,
                                         starting_order = starting_order)
        
    remapped = _get_stateseq_tuples(state_seqs,
                                    dummy_states,
                                    starting_order = starting_order)
    
    dtmp = {
        "dummy_states" : dummy_states,
        "state_map"    : state_map,
        "state_seqs"   : transcode_sequences(remapped, state_map) 
            
    }
    return dtmp
        
def raise_stateseq_orders(state_seqs, reverse_state_map):
    """Map a state sequence from first-order space back to high-order space
    
    Parameters
    ----------    
    state_seqs : list
        List of high-order state sequences
        
    reverse_state_map : dict
        Dictionary mapping first-order model states back to equivalent
        tuples of high-order states. Created by :func:`get_state_mapping`


    Returns
    -------
    list
        List of remapped state sequences, each a :class:`numpy.ndarray`
        
        
    See also
    --------
    lower_stateseq_orders
    """
    return [X[-1] for X in transcode_sequences(state_seqs, reverse_state_map)]
    
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






# def remap_model_parameters(states,
#                            starting_order   = 2,
#                            state_priors     = None,
#                            transition_probs = None,
#                            emission_probs   = None,
#                           ):
#     """Create states and probability tables that map a high-order HMM to an equivalent first-order HMM
#      
#      
#     Parameters
#     ----------
#     states : int
#         Number of model states in high-order space
#      
#     starting_order : int, optional
#         Order of starting HMM/MM (Default: 2)
#          
#     state_priors : class:`numpy.ndarray`, optional
#         Probabilities of starting in any given state
#      
#     transition_probs : :class:`numpy.ndarray`, optional
#         Probabilities of transitions between states
#      
#      
#     Returns
#     -------
#     dict
#         Dictionary of objects describing reduced model:
#          
#             =================  ========================================  =================================================
#             Key                Type                                      Contains
#             -----------------  ----------------------------------------  -------------------------------------------------
#             statemap_forward   :class:`dict`                             Maps tuples of high-order states to new 1st-order
#                                                                          states
#  
#             statemap_reverse   :class:`dict`                             Maps new 1st-order states to corresponding
#                                                                          high-order states
#                                                                         
#             state_priors       :class:`numpy.ndarray`                    Mapping of 2nd order state priors to 1st order
#                                                                          space, with impossible starting points set to
#                                                                          zero
#                                                                         
#             transitions        :class:`numpy.ndarray`                    Mapping of 2nd order transition probabilities
#                                                                          to 1st order space, with impossible transitions
#                                                                          set to zero
#                                                                         
#             emissions          :class:`numpy.ndarray`                    Mapping of 2nd order emisison probabilities to
#                                                                          1st order space
#             =================  ========================================  =================================================
#     """
#     num_starting_states = len(states) 
#     starting_states      = range(num_starting_states)
#      
#     if state_priors is None:
#         state_priors = numpy.full(len(states),1.0/num_starting_states)
#          
#     if transition_probs is None:
#         transition_probs = numpy.full(*([num_starting_states]*(1 + starting_order)),
#                                       1.0/num_starting_states)
#  
#     dtmp = {}    
#     fullstates = copy.deepcopy(states)
#      
#     for n in range(starting_order-1):
#         fullstates.append("start%s" % n)
#         fullstates.append("end%s" % n)
#  
#     statemap_forward, statemap_reverse = map_states(fullstates)
#     dtmp["statemap_forward"] = statemap_forward
#     dtmp["statemap_reverse"] = statemap_reverse
#      
#     num_new_states = len(fullstates)
#      
#     new_state_priors = numpy.zeros(num_new_states, dtype=float)
#     new_transprobs   = numpy.zeros((num_new_states, num_new_states),
#                                    dtype = float)
#      
#     # transition probabilities
#     statepaths = itertools.product( *([starting_states]* (starting_order + 1)))
#     # remap transition probabilities to 1st order space
#     for path in statepaths:
#         old_startstate = path[:-1]
#         old_endstate   = path[1:]
#         new_transprobs[statemap_forward[old_startstate],
#                        statemap_forward[old_endstate]] = transition_probs[path]
#      
#     # add probabilities for added start, end states
#     for i in range(starting_order-1):
#         base = ["start%s" % X for X in range(i)]
#         for remaining in itertools.product( *([starting_states]* starting_order)):
#             idx = tuple(base + list(remaining))
#             fromstate = idx[:-1]
#             tostate   = idx[1:]
#              
#             # MARGINALIZE APPROPRIATELY
#             new_transprobs[statemap_forward[fromstate],
#                            statemap_forward[tostate]] = None 
#      
#     # remap state priors
#     for i in range(num_starting_states):
#         baseidx = ["start%s" % X for X in range(starting_order)]
#         idx = statemap_forward[tuple(baseidx + [i])]
#         new_state_priors[statemap_forward[idx]] = state_priors[i]
#          
#     # remap emissions
#     if emission_probs is not None:        
#         new_emission_probs = [None]*num_new_states
#         for i in range(num_new_states):
#             new_emission_probs[i] = emission_probs[statemap_reverse[i][:-1]]
#         dtmp["emissions"] = new_emission_probs
#      
#     dtmp["state_priors"] = new_state_priors
#     dtmp["transitions"]  = new_transprobs
#     return dtmp