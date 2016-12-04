#!/usr/bin/env python
"""Utilities for changing representations of observation sequences and/or models
"""
import copy
import warnings
import itertools
import numpy



def get_expansion_states(num_states, starting_order=2):
    """Create sorted lists of expansion states required to reduce a model with `num_states` and `starting_order` to an equivalent first-order model
    
    Parameters
    ----------
    num_states : int
        Number of states in high-order HMM/MM
    
    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)
        
        
    Returns
    -------
    list
        List of new start states, sorted
    
    list
        List of new end states, sorted
    """
    newstarts = []
    newends   = []
    for i in range(1, starting_order):
        newend   = - 2*i
        newstart = - 2*i + 1
        newstarts.append(newstart)
        newends.append(newend)
        
    newstarts = list(reversed(newstarts))
    return newstarts, newends
    
def get_state_mapping(num_states, starting_order=2, newstarts=None, newends=None):
    """Create dicts mapping states between a high-order and a first-order HMM,
    adding new start and end states to handle inhomogeneities at sequence ends.
    
    In the representation space of the high-order model, new start states
    are given negative odd indices, and new end states negative even indices.
    
    Parameters
    ----------
    num_states : int
        Number of states in high-order HMM/MM
    
    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)
    
    newstarts : list, optional
        Sorted list of new start states. If `None`, will be calculated using
        :func:`get_expansion_states`

    newends : list, optional
        Sorted list of new start end states. If `None`, will be calculated using
        :func:`get_expansion_states`


    Returns
    -------
    :class:`dict`
        Dictionary mapping old states to new ones
        
    :class:`dict`
        Dictionary mapping new states to old ones
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

    if newstarts is None or newends is None:
        newstarts, newends = get_expansion_states(num_states, starting_order=starting_order)
        
    c = 0
    
    # create maps for newly added start and end states
    for idx in range(starting_order - 1):
        root = newstarts[idx:]
        for symbol in itertools.product(*([states]*(idx + 1))):
            tup = tuple(root + list(symbol))
            forward[tup] = c
            reverse[c]   = tup
            c += 1
            
        root = newends[:len(newends) - idx]
        for symbol in itertools.product(*([states]*(idx + 1))):
            tup = tuple(list(symbol) + root)
            forward[tup] = c
            reverse[c]   = tup
            c += 1
    
    # combinations of true states
    for n, symbol in enumerate(itertools.product(*([states]*starting_order))):
        forward[symbol] = n + c
        reverse[c + n] = symbol
    
    return forward, reverse

def reduce_stateseq_orders(state_seqs,
                           num_states,
                           starting_order = 2,
                           newstarts      = None,
                           newends        = None,
                           state_map      = None):
    """Remap a high-order sequence of states into an equivalent first-order model
    
    Parameters
    ----------
    states : list
        List of states sequences
    
    num_states : int
        Number of states in high-order HMM/MM

    starting_order : int, optional 
        Starting order of HMM/MM (Default: 2)
    
    newstarts : list, optional
        Sorted list of new start states. If `None`, will be calculated using
        :func:`get_expansion_states`

    newends : list, optional
        Sorted list of new start end states. If `None`, will be calculated using
        :func:`get_expansion_states`
        
    state_map : dict, optional
        Dictionary mapping tuples of high-order states to low-order states.
        If `None`, will be calculated useing :func:`get_state_mapping`
    """
    if newstarts is None or newends is None:
        newstarts, newends = get_expansion_states(num_states, starting_order=starting_order)
        
    if state_map is None:
        state_map, _ = get_state_mapping(num_states,
                                         starting_order = starting_order,
                                         newstarts      = newstarts,
                                         newends        = newends)
    
    outseqs = []
    for n, inseq in enumerate(state_seqs):
        if (numpy.array(inseq) < 0).any():
            raise ValueError("Found negative state label in input sequence %s!" % n)
        
        baseseq = newstarts + list(inseq) + newends
        outseqs.append([tuple(baseseq[idx:idx+starting_order]) for idx in range(0, len(baseseq) - starting_order + 1)])
    
    return outseqs 
    


def transcode_sequences(sequences,alphadict):
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


# def reduce_model_order(states,
#                        starting_order   = 2,
#                        state_priors     = None,
#                        transition_probs = None,
#                        emission_probs   = None,
#                       ):
#     """Create states and probability tables that map a high-order HMM to an equivalent 1st-order HMM
#     
#     
#     Parameters
#     ----------
#     states : int
#         number of model states
#     
#     starting_order : int, optional
#         order of starting HMM (Default: 2)
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