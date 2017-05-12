cimport numpy
cimport cython
import numpy
from minihmm.hmm import DiscreteFirstOrderHMM

@cython.boundscheck(False)
def cyvit(object hmm, long [:] emissions):
    cdef:
        int num_states = hmm.num_states
        int emission_length = len(emissions)
        int [:] l = numpy.arange(num_states, dtype=numpy.int32)
        numpy.ndarray T = hmm._logt
        numpy.ndarray E = hmm._loge

        # allocate two holders to prevent repeated reallocation of memory
        int [:, :] state_paths_1 = numpy.full((num_states, emission_length), -1, dtype=numpy.int32)
        int [:, :] state_paths_2 = numpy.full((num_states, emission_length), -1, dtype=numpy.int32)
        int [:, :] state_paths, new_state_paths
        int [:] temp

        numpy.ndarray emission_logprobs
        numpy.ndarray current_probs
        numpy.ndarray prev_probs, new_probs

        int i, j, x, X, final_label

        float total_logprob

    
    state_paths     = state_paths_1
    new_state_paths = state_paths_2
    state_paths[:,0] = l
    
    prev_probs = numpy.array([hmm.state_priors.logprob(X) + E[X, emissions[0]] for X in l])

    j = 1
    while j < emission_length:
        x = emissions[j]
        emission_logprobs = E[:, x]
        current_probs     = T + emission_logprobs[None,:] + prev_probs[:,None]
        
        new_probs = current_probs.max(0)

        for i in l:
            temp = state_paths[<int>current_probs[:, i].argmax(), :]
            new_state_paths[i,:]  = temp
            new_state_paths[i, j + 1] = i

        prev_probs  = new_probs
        state_paths = new_state_paths
        new_state_paths = state_paths_1 if j % 2 == 0 else state_paths_2
        j += 1
    
    final_label   = prev_probs.argmax()
    total_logprob = prev_probs.max()

    dtmp = {
        "viterbi_states"  : numpy.asarray(state_paths[final_label, :]),
        "state_paths"     : numpy.asarray(state_paths),
        "logprob"         : total_logprob,
    }
    return dtmp

