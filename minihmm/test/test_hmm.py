#!/usr/bin/env python
"""Testing functions for FirstOrderHMMs. Viterbi decoding, sequence generation,
the forward-backward algorithm, posterior decoding, and re-training are all tested. 


"""
import unittest
import itertools
import numpy
import scipy.stats

from collections import Counter
from nose.tools import assert_greater_equal, assert_less_equal
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal)

from minihmm.hmm import FirstOrderHMM, DiscreteFirstOrderHMM
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

    model_class = FirstOrderHMM

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
        cls.seq_lengths = [5,10] #,20]

        cls.test_obs_seqs   = []    # test observation sequences
        cls.test_state_seqs = []    # state paths through test observation seqs. not currently used
        cls.decode_tests    = []    # tests for Viterbi and posterior decoding

        cls.expected_forward_logprobs = [] # log probabilities for cls.test_obs_seqs
        cls.found_forward_logprobs    = []
        cls.found_forward_scalefactors      = []
        cls.found_forward_scaled_forward_matrices = []

        cls.expected_joint_logprobs = [] # expected joint probabilities for states and observations

        print("Setting up class %s" % cls.__name__)
        cls.do_subclass_setup()
        hmm = cls.model_class(**cls.models["generating"])
        cls.generating_hmm = hmm
        cls.naive_hmm      = FirstOrderHMM(**cls.models["naive"])

        numpy.random.seed(_TRAINING_SEED)

        for x in cls.seq_lengths:
            states, obs, _ = hmm.generate(x)
            cls.test_state_seqs.append(states)
            cls.test_obs_seqs.append(obs)

        # precalculate useful quantities on observation sequences
        for obs in cls.test_obs_seqs:
            # run forward algorithm; we'll use their results for tests below
            logprob, scaled_forward, scaled_backward, scale_factors, ksi = hmm.forward_backward(obs)
            cls.found_forward_logprobs.append(logprob)
            cls.found_forward_scalefactors.append(scale_factors)
            cls.found_forward_scaled_forward_matrices.append(scaled_forward)

            # manually calculate probabilities of generated observations via dynamic programming
            # the forward probability of an observation sequence is equal to the sum of its joint 
            # probabilities with all possible state paths. We calculate each joint probability here,
            # then sum these to find the total expected probability
            my_states = list(range(hmm.num_states))
            paths = { (X,) : hmm.state_priors.logprob(X) + hmm.emission_probs[X].logprob(obs[0]) for X in my_states }

            for my_obs in obs[1:]:
                new_paths = {}
                for partial, prob in paths.items():
                    for new_state in my_states:
                        new_paths[ tuple(list(partial) + [new_state])] = prob + hmm.trans_probs.logprob(partial[-1], new_state) + hmm.emission_probs[new_state].logprob(my_obs)
                        
                paths = new_paths

            dp_prob = numpy.log(numpy.exp(paths.values()).sum())
            cls.expected_forward_logprobs.append(dp_prob)
            cls.expected_joint_logprobs.append(paths)

        # Generate test cases for Viterbi and posterior decoding            
        cls.decode_tests = [hmm.generate(1000) for _ in range(10)]
            
        print("Set up class %s" % cls.__name__)

    @unittest.skip
    def test_generate(self):
        # testable
        
        # 1. test length of state and observation sequences is correct
        # 2. test sampling is according to joint distribution?
        # TODO: what else?
        assert False

    def test_sample(self):
        # Test sampling algorithm by checking the slope and intercept of the regression line
        # between expected and observed numbers of observations for each state path
        num_samples = 10000

        for n, (obs, logprob, joint_probs) in enumerate(zip(self.test_obs_seqs,
                                                           self.expected_forward_logprobs,
                                                           self.expected_joint_logprobs)):
            cond_probs = { K : V - logprob for K,V in joint_probs.items() }
            paths    = self.generating_hmm.sample(obs, num_samples=num_samples)
            samples  = Counter([tuple(X.astype(int)) for X in paths])
            expected = numpy.zeros(len(cond_probs))
            found    = numpy.zeros(len(cond_probs))

            for n, (k, v) in enumerate(sorted(cond_probs.items())):
                expected[n] = num_samples * numpy.exp(v)
                found[n]    = samples.get(k,0)

            m, b, r, p, std = scipy.stats.linregress(expected,found)
            assert_less_equal(abs(m - 1), 0.05, "Slope '%s' further from 1.0 than expected." % (m))
            assert_less_equal(abs(b), 1, "Intercept '%s' further from 0.0 than expected." % (b))
        
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
        for n, (expected, found) in enumerate(zip(self.expected_forward_logprobs, self.found_forward_logprobs)):
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
        for n, (obs, expected) in enumerate(zip(self.test_obs_seqs, self.expected_forward_logprobs)):
            found = self.generating_hmm.fast_forward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (n,
                                                                                                 self.name,
                                                                                                 expected,
                                                                                                 found,
                                                                                                 abs(expected-found)
                                                                                                )
            yield assert_almost_equal, expected, found, 7, msg

    def test_scaled_forward_is_scaled_to_one(self):
        # n.b. at present we're only scaling from timestep 1 onwards
        for n, scaled_forward in enumerate(self.found_forward_scaled_forward_matrices):
            yield assert_almost_equal, scaled_forward[1:].sum(1), 1

    def test_forward_backward_scalefactors_product_sum_is_consistent(self):
        # product of scale factors and forward algorithm at each timestep
        # should equal probability of that sequence up to that point

        # n.b. at present we're only scaling from timestep 1 onwards
        for n, (obs, scale_factors, scaled_forward) in enumerate(zip(self.test_obs_seqs,
                                                                     self.found_forward_scalefactors,
                                                                     self.found_forward_scaled_forward_matrices)):

            expected = [numpy.exp(self.generating_hmm.fast_forward(obs[:X+1])) for X in range(len(obs))]
            found    = scaled_forward.sum(1) * scale_factors.cumprod()
            yield assert_almost_equal, expected[1:], found[1:]

    def test_forward_backward_logprob(self):
        # test forward algorithm portion of forward_backward
        
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected_logprob) in enumerate(zip(self.test_obs_seqs, self.expected_forward_logprobs)):
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

    def test_joint_path_logprob(self):
        assert False

    def test_conditional_path_logprob(self):
        assert False

    def test_forward_backward_backward(self):
        # TODO: test backward component of forward-backward algorithm
        assert False

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
        cls.min_frac_equal = 0.69

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



class TestA2DiscreteCoin(TestACoin):

    model_class = DiscreteFirstOrderHMM

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Coin example"
        cls.min_frac_equal = 0.69

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
                "emission_probs"   : MatrixFactor(numpy.array([[0.6,0.4],
                                                              [0.15,0.85]])),
            },
            "naive"      : {
                "state_priors"     : ArrayFactor([0.48,0.52]),
                "trans_probs"      : MatrixFactor(numpy.array([
                                                    [0.5, 0.5],
                                                    [0.5, 0.5]])),
                "emission_probs"   : MatrixFactor(numpy.array([[0.5,0.5],
                                                               [0.5,0.5]])),
             }
        }

    @classmethod
    def setUpClass(cls):
        cls.seq_lengths = [5,10] #,20]

        cls.test_obs_seqs   = []    # test observation sequences
        cls.test_state_seqs = []    # state paths through test observation seqs. not currently used
        cls.decode_tests    = []    # tests for Viterbi and posterior decoding

        cls.expected_forward_logprobs = [] # log probabilities for cls.test_obs_seqs
        cls.found_forward_logprobs    = []
        cls.found_forward_scalefactors      = []
        cls.found_forward_scaled_forward_matrices = []

        cls.expected_joint_logprobs = [] # expected joint probabilities for states and observations

        print("Setting up class %s" % cls.__name__)
        cls.do_subclass_setup()
        hmm = cls.model_class(**cls.models["generating"])
        cls.generating_hmm = hmm
        cls.naive_hmm      = FirstOrderHMM(**cls.models["naive"])

        numpy.random.seed(_TRAINING_SEED)

        for x in cls.seq_lengths:
            states, obs, _ = hmm.generate(x)
            cls.test_state_seqs.append(states)
            cls.test_obs_seqs.append(obs)

        # precalculate useful quantities on observation sequences
        for obs in cls.test_obs_seqs:
            # run forward algorithm; we'll use their results for tests below
            logprob, scaled_forward, scaled_backward, scale_factors, ksi = hmm.forward_backward(obs)
            cls.found_forward_logprobs.append(logprob)
            cls.found_forward_scalefactors.append(scale_factors)
            cls.found_forward_scaled_forward_matrices.append(scaled_forward)

            # manually calculate probabilities of generated observations via dynamic programming
            # the forward probability of an observation sequence is equal to the sum of its joint 
            # probabilities with all possible state paths. We calculate each joint probability here,
            # then sum these to find the total expected probability
            my_states = list(range(hmm.num_states))
            paths = { (X,) : hmm.state_priors.logprob(X) + hmm._loge[X,obs[0]] for X in my_states }

            for my_obs in obs[1:]:
                new_paths = {}
                for partial, prob in paths.items():
                    for new_state in my_states:
                        new_paths[ tuple(list(partial) + [new_state])] = prob + hmm.trans_probs.logprob(partial[-1], new_state) + hmm._loge[new_state,my_obs]
                        
                paths = new_paths

            dp_prob = numpy.log(numpy.exp(paths.values()).sum())
            cls.expected_forward_logprobs.append(dp_prob)
            cls.expected_joint_logprobs.append(paths)

        # Generate test cases for Viterbi and posterior decoding            
        cls.decode_tests = [hmm.generate(1000) for _ in range(10)]
            
        print("Set up class %s" % cls.__name__)



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
                "trans_probs"    : MatrixFactor(numpy.full((4,4), 1.0/16, dtype=float)),
                "state_priors"   : ArrayFactor([0.25,0.25,0.25,0.25]),
                "emission_probs" : [
                      ScipyDistributionFactor(scipy.stats.poisson,3),
                      ScipyDistributionFactor(scipy.stats.poisson,2),
                      ScipyDistributionFactor(scipy.stats.poisson,1),
                      ScipyDistributionFactor(scipy.stats.poisson,4),
                 ]
            }
        }
