
import numpy
cimport numpy

cdef class CyDHMM:

    def __init__(self,
                 numpy.ndarray state_priors,
                 numpy.ndarray emission_probs,
                 numpy.ndarray trans_probs):

        self.state_priors   = state_priors
        self.emission_probs = emission_probs
        self.trans_probs    = trans_probs

        self._logp = numpy.log(state_priors)
        self._logt = numpy.log(trans_probs)
        self._loge = numpy.log(emission_probs)

    cdef c_forward(self, numpy.ndarray emissions):
        cdef:
            int [:] my_view = emissions
            float total_logprob
            float [:] scale_factors   = numpy.full(numpy.nan, emissions.shape[0])
            float [:] scaled_forward  



