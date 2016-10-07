#!/usr/bin/env python
"""Testing functions"""
import numpy
import scipy.stats
import sys

from nose.tools import assert_greater_equal
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal)

from minihmm.hmm import FirstOrderHMM
from minihmm.estimators import (UnivariateGaussianEmissionEstimator,
                                DiscreteStatePriorEstimator,
                                DiscreteEmissionEstimator,
                                DiscreteTransitionEstimator,
                                PseudocountTransitionEstimator,
                                )
from minihmm.training import train_baum_welch, neg_exp_noise_gen
from minihmm.factors import (ArrayFactor,
                             MatrixFactor,
                             LogFunctionFactor,
                             ScipyDistributionFactor)


#_casino_flat_params    = [0.5,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,0.5,0.5]
#_gaussian_flat_params  = [0.5,0,1,0,1,0.5,0.5]
#_gaussian_flat_params2 = [0.5,-1,1,1,1,0.5,0.5]

# sample HMMs ------------------------------------------------------------------

_FORWARD_SEED  = 5139284
_TRAINING_SEED = 134067



class _BaseExample():

    # define these variables in this method in subclasses    
    @classmethod
    def do_subclass_setup(cls):
        cls.name = ""
        cls.min_frac_equal = 0.8
        cls.state_prior_estimator = None
        cls.transition_estimator  = None
        cls.emission_estimator    = None
        cls.models = {
            "generating" : {
                "trans_probs"      : None,
                "state_priors"     : None,
                "emission_probs"   : None,
            },
            "naive"      : {
                "trans_probs"      : None,
                "state_priors"     : None,
                "emission_probs"   : None,
            }
        }

    @classmethod
    def setUpClass(cls):
        cls.do_subclass_setup()
        cls.generating_hmm = FirstOrderHMM(**cls.models["generating"])
        cls.naive_hmm      = FirstOrderHMM(**cls.models["naive"])

        numpy.random.seed(_TRAINING_SEED)
        (cls.states,
         cls.observations,
         cls.logprobs) = zip(*[cls.generating_hmm.generate(200) for _ in range(500)])

    def test_generate(self):
        pass

    def test_viterbi(self):
        for expected_states, obs in zip(self.states, self.observations):
            found_states = self.generating_hmm.viterbi(obs)["viterbi_states"]
            frac_equal = 1.0 * (expected_states == found_states).sum() / len(expected_states)
            yield assert_greater_equal, frac_equal, self.min_frac_equal, msg
         

    def test_posterior_decode(self):
        pass

    def test_forward_logprob(self):
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected) in enumerate(zip(self.observations, self.logprobs)):
            found, scaled_forward, b, scale_factors = self.hmm.forward(obj)
            msg = "Failed test case %s on HMM %s. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                              self.name,
                                                                                              expected,
                                                                                              found,
                                                                                              abs(expected-found)
                                                                                              )
            yield assert_almost_equal, expected, found, msg

    def test_fast_forward(self):
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected) in enumerate(zip(self.observations, self.logprobs)):
            found = self.hmm.fast_forward(obj)
            msg = "Failed test case %s on HMM %s. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                              self.name,
                                                                                              expected,
                                                                                              found,
                                                                                              abs(expected-found)
                                                                                              )
            yield assert_almost_equal, expected, found, msg

    def test_fast_forward(self):
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected) in enumerate(zip(self.observations, self.logprobs)):
            found = self.hmm.logprob(obj)
            msg = "Failed test case %s on HMM %s. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                              self.name,
                                                                                              expected,
                                                                                              found,
                                                                                              abs(expected-found)
                                                                                              )
            yield assert_almost_equal, expected, found, msg

    def test_forward_backward(self):
        pass

    def test_train(self):
        mdict = train_baum_welch(self.naive_hmm,
                                 self.observations,
                                 state_prior_estimator = self.state_prior_estimator,
                                 transition_estimator = self.transition_estimator,
                                 emission_estimator   = self.emission_estimator,
                                 noise_weights = neg_exp_noise_gen(),
#                                 miniter = 200,
#                                 pseudocount_weights  = iter([0]),
                                 )
        new_model = mdict["best_model"]
        print(mdict)

        yield assert_array_almost_equal, self.generating_hmm.trans_probs.data, new_model.trans_probs.data
        for expected, found in zip(self.generating_hmm.emission_probs, new_model.emission_probs):
            if expected is not None:
                yield assert_array_almost_equal, expected.data, found.data



class TestCoinExample(_BaseExample):

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Coin example"
        cls.min_frac_equal = 0.7

        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator  = DiscreteTransitionEstimator()
        cls.emission_estimator    = DiscreteEmissionEstimator(2)
 
#        cls.transition_estimator = PseudocountTransitionEstimator(numpy.array([
#            [ 0 ,  0,  1,  1],
#            [ 0, 1.0,  0,  0],
#            [ 0,   0,  1,  1],
#            [ 0,   0,  1,  1]
#            ]))

        cls.models = {
            "generating" : {
                "state_priors"     : ArrayFactor([0.0,1.0]),
                "trans_probs"      : MatrixFactor(numpy.array([
                                                    [0.7, 0.3],
                                                    [0.3, 0.7]])),
                "emission_probs"   : [ArrayFactor([0.5,0.5]),
                                      ArrayFactor([0.2,0.8])],
            },
            "naive"      : {
                "state_priors"     : ArrayFactor([0.48,0.52]),
                "trans_probs"      : MatrixFactor(numpy.array([
                                                    [0.5, 0.5],
                                                    [0.5, 0.5]])),
                "emission_probs"   : [ArrayFactor([0.5,0.5]),
                                      ArrayFactor([0.5,0.5])],
             }
        }



class TestGaussianExample(_BaseExample):

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Gaussian example"
        cls.min_frac_equal = 0.7

        transitions = numpy.matrix([[0.9,0.1],
                                    [0.25,0.75]])

        cls.models = {
            "generating" : {
                "trans_probs"    : MatrixFactor(transitions),
                "state_priors"   : ArrayFactor([0.8,0.2]),
                "emission_probs" : [ScipyDistributionFactor(scipy.stats.norm,loc=0,scale=0.5),
                                    ScipyDistributionFactor(scipy.stats.norm,loc=5,scale=10)],
            },
            "naive"      : {
                "emission_probs" : [ScipyDistributionFactor(scipy.stats.norm,loc=0,scale=0.6),
                                    ScipyDistributionFactor(scipy.stats.norm,loc=2,scale=1),
                                   ],
                "state_priors"   : ArrayFactor([0.48,0.52]),
                "trans_probs"    : MatrixFactor([[0.5, 0.5], 
                                                 [0.5, 0.5]]),
            }
        }

    


def get_coins(hmm_type=FirstOrderHMM):
    """Constrruct a two-state HMM with fair and unfair coins

    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    # 0 is fair coin
    # 1 is unfair coin
    #
    # for emissions, 0 is heads, 1 is tails
    emission_factors = [None,
                        None,
                        ArrayFactor([0.5,0.5]),
                        ArrayFactor([0.2,0.8])
                        ]
    trans_probs = MatrixFactor(numpy.array([[0,  0,   1  , 0.0],
                                            [0,  0,   0,   0.0],
                                            [0,  0.1, 0.7, 0.2],
                                            [0,  0.1, 0.2, 0.7]]))
    return hmm_type(emission_factors,trans_probs)


#def get_fourstate(hmm_type=FirstOrderHMM):
#    """Construct a four-state HMM over discrete values for testing purposes
#    
#    This also implicitly tests ArrayFactor, MatrixFactor, and ScipyDistributionFactor
#
#    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
#                              or a subclass)
#    """
#    import scipy.stats
#    state_priors = ArrayFactor([0.25,0.25,0.25,0.25])
#    trans_probs = MatrixFactor(numpy.matrix([[0.8,0.05,0.05,0.1],
#                                            [0.2,0.6,0.1,0.1],
#                                            [0.01,0.97,0.01,0.01],
#                                            [0.45,0.01,0.04,0.5],
#                                            ]))
#    emission_probs = [ScipyDistributionFactor(scipy.stats.poisson,1),
#                      ScipyDistributionFactor(scipy.stats.poisson,5),
#                      ScipyDistributionFactor(scipy.stats.poisson,10),
#                      ScipyDistributionFactor(scipy.stats.poisson,25),
#                      ]
#    return hmm_type(state_priors,emission_probs,trans_probs)
#
#
## testing functions ------------------------------------------------------------
