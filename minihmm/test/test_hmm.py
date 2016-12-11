#!/usr/bin/env python
"""Testing functions for FirstOrderHMMs. Viterbi decoding, sequence generation,
the forward-backward algorithm, posterior decoding, and re-training are all tested. 


"""
import unittest
import itertools
import numpy
import scipy.stats
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
        print("Setting up class %s" % cls.__name__)
        cls.do_subclass_setup()
        cls.generating_hmm = FirstOrderHMM(**cls.models["generating"])
        cls.naive_hmm      = FirstOrderHMM(**cls.models["naive"])

        print("    Generating sequences ...")
        numpy.random.seed(_TRAINING_SEED)
        cls.forward_tests = [cls.generating_hmm.generate(X) for X in [5,10]]#,20]])#,150]])

        cls.expected_forward_logprobs = []          
        # evaluate probabilities of generated observations by brute force
        # to compare against forward algorithm calculations later
        print("    Calculating probabilities ...")
        for _,  obs_seq, _ in cls.forward_tests:
            state_paths = itertools.product(list(range(cls.generating_hmm.num_states)),
                                            repeat=len(obs_seq))
            total_prob  = 0
            for my_path in state_paths:
                my_logprob = []
                last_state = my_path[0]
                my_logprob.append(cls.generating_hmm.state_priors.logprob(last_state))
                my_logprob.append(cls.generating_hmm.emission_probs[last_state].logprob(obs_seq[0]))
                for my_state, my_obs in zip(my_path, obs_seq)[1:]:
                    my_logprob.append(cls.generating_hmm.trans_probs.logprob(last_state,my_state))
                    my_logprob.append(cls.generating_hmm.emission_probs[my_state].logprob(my_obs))
                    last_state = my_state
                
                total_prob += numpy.exp(numpy.nansum(my_logprob))
            
            cls.expected_forward_logprobs.append(numpy.log(total_prob))

        # Generate test cases for Viterbi and posterior decoding            
        print("Creating decoding testss")
        cls.decode_tests = [cls.generating_hmm.generate(1000) for _ in range(10)]
            
        print("Set up class %s" % cls.__name__)

    @unittest.skip
    def test_generate(self):
        # TODO : not sure what proper test is. Median probability should be 
        # less than mean, if 
        assert False

    @unittest.skip
    def test_sample(self):
        # TODO : not sure what proper test is; distribution of samples
        # should approximate distribution of HMM, but we don't know what
        # distribution of HMM actually is.
        
        #
        # Maybe calculate ML solution (Viterbi) and make sure everything sampled
        # is lower probability?
        #
        # Some test of shape of distribution? Require one mode? 
        assert False
        
    def test_viterbi(self):
        # make sure viterbi calls are above accuracy threshold listed above 
        for expected_states, obs, _ in self.decode_tests:
            found_states = self.generating_hmm.viterbi(obs)["viterbi_states"]
            frac_equal = 1.0 * (expected_states == found_states).sum() / len(expected_states)
            msg = "Failed viterbi test for test case '%s'. Expected at least %s%% accuracy. Got %s%%." % (self.name,self.min_frac_equal,frac_equal)
            assert_greater_equal(frac_equal, self.min_frac_equal, msg)
         
    def test_posterior_decode(self):
        # make sure posterior decode calls are above accuracy threshold listed above
        for expected_states, obs, _ in self.decode_tests:
            found_states = self.generating_hmm.posterior_decode(obs)[0]
            frac_equal = 1.0 * (expected_states == found_states).sum() / len(expected_states)
            msg = "Failed posterior decode test for test case '%s'. Expected at least %s%% accuracy. Got %s%%." % (self.name,self.min_frac_equal,frac_equal)
            assert_greater_equal(frac_equal, self.min_frac_equal, msg)

    def test_forward_logprob(self):
        # make sure vectorized forward probability calculations match those calced by brute force
        numpy.random.seed(_FORWARD_SEED)
        for n, ((_,obs,_), expected) in enumerate(zip(self.forward_tests, self.expected_forward_logprobs)):
            found, _, _ = self.generating_hmm.forward(obs)
            msg = "Failed forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                                  self.name,
                                                                                                  expected,
                                                                                                  found,
                                                                                                  abs(expected-found)
                                                                                                 )
            yield assert_almost_equal, expected, found, 7, msg

    def test_fast_forward(self):
        # make sure fast forward probability calculations match those calced by brute force
        numpy.random.seed(_FORWARD_SEED)
        for n, ((_,obs,_), expected) in enumerate(zip(self.forward_tests, self.expected_forward_logprobs)):
            found = self.generating_hmm.fast_forward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                                 self.name,
                                                                                                 expected,
                                                                                                 found,
                                                                                                 abs(expected-found)
                                                                                                )
            yield assert_almost_equal, expected, found, 7, msg

    def test_forward_backward(self):
        # test forward algorithm portion of forward_backward
        
        # TODO: test backward component
        numpy.random.seed(_FORWARD_SEED)
        for n, ((_,obs,_), expected_logprob) in enumerate(zip(self.forward_tests, self.expected_forward_logprobs)):
            (found_logprob,
             scaled_forward,
             scaled_backward,
             scale_factors,
             ksi)     = self.generating_hmm.forward_backward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                                 self.name,
                                                                                                 expected_logprob,
                                                                                                 found_logprob,
                                                                                                 abs(expected_logprob-found_logprob)
                                                                                                )
            yield assert_almost_equal, expected_logprob, found_logprob, 7, msg

#     def test_train(self):
#         mdict = train_baum_welch(self.naive_hmm,
#                                  self.observations,
#                                  state_prior_estimator = self.state_prior_estimator,
#                                  transition_estimator = self.transition_estimator,
#                                  emission_estimator   = self.emission_estimator,
#                                  noise_weights = neg_exp_noise_gen(),
# #                                 miniter = 200,
# #                                 pseudocount_weights  = iter([0]),
#                                  )
#         new_model = mdict["best_model"]
#         print(mdict)
# 
#         yield assert_array_almost_equal, self.generating_hmm.trans_probs.data, new_model.trans_probs.data
#         for expected, found in zip(self.generating_hmm.emission_probs, new_model.emission_probs):
#             if expected is not None:
#                 yield assert_array_almost_equal, expected.data, found.data



class TestACoin(_BaseExample):

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
                "state_priors"     : ArrayFactor([0.005,0.995]),
                "trans_probs"      : MatrixFactor(numpy.array([
                                                    [0.8, 0.2],
                                                    [0.3, 0.7]])),
                "emission_probs"   : [ArrayFactor([0.6,0.4]),
                                      ArrayFactor([0.15,0.85])],
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



class TestBTwoGaussian(_BaseExample):

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Two gaussian example"
        cls.min_frac_equal = 0.7

        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator  = DiscreteTransitionEstimator()
        cls.emission_estimator    = UnivariateGaussianEmissionEstimator()

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

    

class TestCFourPoisson(_BaseExample):

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Four-state Poisson example"
        cls.min_frac_equal = 0.7

        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator  = DiscreteTransitionEstimator()
        
        # FIXME
        cls.emission_estimator    = UnivariateGaussianEmissionEstimator()

        transitions = numpy.matrix([[0.8,0.05,0.05,0.1],
                                    [0.2,0.6,0.1,0.1],
                                    [0.01,0.97,0.01,0.01],
                                    [0.45,0.01,0.04,0.5],
                                   ])

        cls.models = {
            "generating" : {
                "trans_probs"    : MatrixFactor(transitions),
                "state_priors"   : ArrayFactor([0.7,0.05,0.15,0.10]),
                "emission_probs" : [
                      ScipyDistributionFactor(scipy.stats.poisson,1),
                      ScipyDistributionFactor(scipy.stats.poisson,5),
                      ScipyDistributionFactor(scipy.stats.poisson,10),
                      ScipyDistributionFactor(scipy.stats.poisson,25),
                 ]

            },
            "naive"      : {
                "trans_probs"    : MatrixFactor(numpy.full((4,4),1.0/16)),
                "state_priors"   : ArrayFactor([0.25,0.25,0.25,0.25]),
                "emission_probs" : [
                      ScipyDistributionFactor(scipy.stats.poisson,3),
                      ScipyDistributionFactor(scipy.stats.poisson,2),
                      ScipyDistributionFactor(scipy.stats.poisson,1),
                      ScipyDistributionFactor(scipy.stats.poisson,4),
                 ]
            }
        }