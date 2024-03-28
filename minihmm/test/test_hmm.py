#!/usr/bin/env python
"""Testing functions for FirstOrderHMMs. Viterbi decoding, sequence generation,
the forward-backward algorithm, posterior decoding, and re-training are all tested. 


"""
import unittest
import itertools
import warnings
import pickle

import numpy
import scipy.stats

import jsonpickle
import jsonpickle.ext.numpy
jsonpickle.ext.numpy.register_handlers()

from collections import Counter

from nose.plugins.attrib import attr
from nose.tools import (
    assert_greater_equal,
    assert_less_equal,
    assert_dict_equal,
    assert_list_equal,
    assert_almost_equal,
)
from numpy.testing import (assert_array_equal, assert_array_almost_equal, assert_almost_equal)

from minihmm.hmm import FirstOrderHMM
from minihmm.estimators import (
    UnivariateGaussianEmissionEstimator,
    DiscreteStatePriorEstimator,
    DiscreteEmissionEstimator,
    DiscreteTransitionEstimator,
    PseudocountTransitionEstimator,
)
from minihmm.training import train_baum_welch, neg_exp_noise_gen
from minihmm.factors import (ArrayFactor, MatrixFactor, LogFunctionFactor, ScipyDistributionFactor)

from minihmm.test.common import (
    check_array_equal,
    check_equal,
    get_fourstate_poisson,
)

_FORWARD_SEED = 5139284
_TRAINING_SEED = 134067


class _BaseExample():

    # define these variables in this method in subclasses
    @classmethod
    def do_subclass_setup(cls):

        # name of test suite
        cls.name = ""

        # minimum fraction expected to be equal in viterbi decoding test
        cls.min_frac_equal = 0.8

        # dict representation of hmm
        cls.hmm_dict = None

        # HMM instance used for testing
        cls.generating_hmm = None

    @classmethod
    def setUpClass(cls):
        # Note- 10 is quite slow for multiple tests below, given that the
        # checks use brute-force implementations to verify the dynamic
        # programming implementation used in miniHMM.
        cls.seq_lengths = [5, 10]  #,20]

        cls.test_obs_seqs = []  # test observation sequences
        cls.test_state_seqs = []  # state paths through test observation seqs. not currently used
        cls.decode_tests = []  # tests for Viterbi and posterior decoding

        cls.expected_forward_logprobs = []  # log probabilities for cls.test_obs_seqs
        cls.found_forward_logprobs = []
        cls.found_forward_scalefactors = []
        cls.found_forward_scaled_forward_matrices = []

        cls.expected_joint_logprobs = []  # expected joint probabilities for states and observations

        print("Setting up class %s" % cls.__name__)
        cls.do_subclass_setup()
        hmm = cls.generating_hmm
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
            paths = {
                (X, ): hmm.state_priors.logprob(X) + hmm.emission_probs[X].logprob(obs[0])
                for X in my_states
            }

            for my_obs in obs[1:]:
                new_paths = {}
                for partial, prob in paths.items():
                    for new_state in my_states:
                        new_paths[tuple(list(partial) + [new_state])
                                  ] = prob + hmm.trans_probs.logprob(
                                      partial[-1], new_state
                                  ) + hmm.emission_probs[new_state].logprob(my_obs)

                paths = new_paths

            dp_prob = numpy.log(numpy.exp(list(paths.values())).sum())
            cls.expected_forward_logprobs.append(dp_prob)
            cls.expected_joint_logprobs.append(paths)

        # Generate test cases for Viterbi and posterior decoding
        cls.decode_tests = [hmm.generate(1000) for _ in range(10)]

        cls.from_pickle = pickle.loads(pickle.dumps(cls.generating_hmm))
        cls.from_json = jsonpickle.decode(jsonpickle.encode(cls.generating_hmm))

        print("Set up class %s" % cls.__name__)

    # override in subclass
    @unittest.skip
    def test_to_json(self):
        assert False

    def test_from_json(self):
        gen = self.generating_hmm
        rev = self.from_json
        yield check_array_equal, gen.state_priors.data, rev.state_priors.data
        yield check_array_equal, gen.trans_probs.data, rev.trans_probs.data
        for i in range(gen.num_states):
            yield check_equal, gen.emission_probs[i], rev.emission_probs[i]

    @attr("slow")
    def test_logprob(self):
        # make sure fast forward probability calculations match those calced by brute force
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected) in enumerate(zip(self.test_obs_seqs,
                                                self.expected_forward_logprobs)):
            found = self.generating_hmm.fast_forward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (
                n, self.name, expected, found, abs(expected - found)
            )
            yield assert_almost_equal, expected, found, 7, msg

    def test_fast_forward(self):
        # make sure fast forward probability calculations match those calced by brute force
        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected) in enumerate(
            zip(self.test_obs_seqs, self.expected_forward_logprobs)
        ):
            found = self.generating_hmm.fast_forward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (
                n, self.name, expected, found, abs(expected - found)
            )
            yield assert_almost_equal, expected, found, 7, msg

    def test_forward_logprob(self):
        # make sure vectorized forward probability calculations match those calced by brute force
        numpy.random.seed(_FORWARD_SEED)
        for n, (expected, found) in enumerate(
            zip(self.expected_forward_logprobs, self.found_forward_logprobs)
        ):
            msg = "Failed forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (
                n, self.name, expected, found, abs(expected - found)
            )
            yield assert_almost_equal, expected, found, 7, msg

    def test_forward_backward_scaled_forward_set_to_one(self):
        # n.b. at present we're only scaling from timestep 1 onwards
        for n, scaled_forward in enumerate(self.found_forward_scaled_forward_matrices):
            yield assert_almost_equal, scaled_forward[1:].sum(1), 1

    def test_forward_scale_factors_product_sum_is_consistent(self):
        # product of scale factors and forward algorithm at each timestep
        # should equal probability of that sequence up to that point
        #
        # this test is misleading if fast_forward() is wrong

        # n.b. at present we're only scaling from timestep 1 onwards
        for n, obs in enumerate(self.test_obs_seqs):
            _, scaled_forward, scale_factors = self.generating_hmm.forward(obs)
            expected = [
                numpy.exp(self.generating_hmm.fast_forward(obs[:X + 1])) for X in range(len(obs))
            ]
            found = scaled_forward.sum(1) * scale_factors.cumprod()
            yield assert_almost_equal, expected[1:], found[1:]

    def test_forward_backward_logprob(self):
        # test forward algorithm portion of forward_backward
        # make sure forward coefficients match those calculated by brute force
        # at each timestep

        numpy.random.seed(_FORWARD_SEED)
        for n, (obs, expected_logprob) in enumerate(zip(self.test_obs_seqs,
                                                        self.expected_forward_logprobs)):
            (found_logprob, scaled_forward, scaled_backward, scale_factors,
             ksi) = self.generating_hmm.forward_backward(obs)
            msg = "Failed fast_forward test case '%s' on HMM '%s'. Expected: '%s'. Found '%s'. Diff: '%s'." % (
                n, self.name, expected_logprob, found_logprob,
                abs(expected_logprob - found_logprob)
            )
            yield assert_almost_equal, expected_logprob, found_logprob, 7, msg

    def test_forward_backward_scalefactors_product_sum_is_consistent(self):
        # product of scale factors and forward algorithm at each timestep
        # should equal probability of that sequence up to that point
        #
        # this test is misleading if fast_forward() is wrong

        # n.b. at present we're only scaling from timestep 1 onwards
        for n, (obs, scale_factors, scaled_forward) in enumerate(
                zip(self.test_obs_seqs, self.found_forward_scalefactors,
                    self.found_forward_scaled_forward_matrices)):

            expected = [
                numpy.exp(self.generating_hmm.fast_forward(obs[:X + 1])) for X in range(len(obs))
            ]
            found = scaled_forward.sum(1) * scale_factors.cumprod()
            yield assert_almost_equal, expected[1:], found[1:]

    # override in subclass
    @unittest.skip
    def test_forward_backward_backward(self):
        # TODO: test backward component of forward-backward algorithm
        # this is currently tested implicitly, as the tests in test_train.py
        # pass
        assert False

    @attr("slow")
    def test_posterior_decode_path_accuracy(self):
        # make sure posterior decode calls are above accuracy threshold listed above
        for expected_states, obs, _ in self.decode_tests:
            found_states = self.generating_hmm.posterior_decode(obs)[0]
            frac_equal = 1.0 * (expected_states == found_states).sum() / len(expected_states)
            msg = "Failed posterior decode test for test case '%s'. Expected at least %s%% accuracy. Got %s%%." % (
                self.name, self.min_frac_equal, frac_equal
            )
            assert_greater_equal(frac_equal, self.min_frac_equal, msg)

    # override in subclass
    @unittest.skip
    def test_generate(self):
        # FIXME: implement this isolated test.
        #
        #
        # generate() is currently tested implicitly, as retraining on generated
        # sequences in test_train.py actually converges.
        #
        #
        # 1. test length of state and observation sequences is correct
        # 2. test sampling is according to joint distribution
        assert False

    @attr("slow")
    def test_joint_path_logprob(self):
        for n, (obs, expected_joint_probs) in enumerate(zip(self.test_obs_seqs,
                                                            self.expected_joint_logprobs)):
            for path, path_prob in expected_joint_probs.items():
                found_joint_prob = self.generating_hmm.joint_path_logprob(path, obs)
                assert_almost_equal(found_joint_prob, path_prob)

    def test_conditional_path_logprob(self):
        for n, (obs, expected_joint_probs) in enumerate(zip(self.test_obs_seqs,
                                                            self.expected_joint_logprobs)):
            total_logprob = self.generating_hmm.fast_forward(obs)
            for path, path_prob in expected_joint_probs.items():
                found_cond_prob = self.generating_hmm.conditional_path_logprob(path, obs)
                assert_almost_equal(found_cond_prob, path_prob - total_logprob)

    @attr("slow")
    def test_sample(self):
        # Test sampling algorithm by checking the slope and intercept of the regression line
        # between expected and observed numbers of observations for each state path
        num_samples = 10000

        for n, (obs, logprob, joint_probs) in enumerate(zip(
                self.test_obs_seqs, self.expected_forward_logprobs, self.expected_joint_logprobs)):
            cond_probs = {K: V - logprob for K, V in joint_probs.items()}
            paths = self.generating_hmm.sample(obs, num_samples=num_samples)
            samples = Counter([tuple(X.astype(int)) for X in paths])
            expected = numpy.zeros(len(cond_probs))
            found = numpy.zeros(len(cond_probs))

            for n, (k, v) in enumerate(sorted(list(cond_probs.items()))):
                expected[n] = num_samples * numpy.exp(v)
                found[n] = samples.get(k, 0)

            m, b, r, p, std = scipy.stats.linregress(expected, found)
            assert_less_equal(abs(m - 1), 0.02, "Slope '%s' further from 1.0 than expected." % (m))
            assert_less_equal(abs(b), 1.0, "Intercept '%s' further from 0.0 than expected." % (b))
            assert_greater_equal(r, 0.95, "r '%s' less than 0.95 than expected." % r)

    @attr("slow")
    def test_viterbi_path_accuracy(self):
        # make sure viterbi calls are above accuracy threshold listed above
        # with ground-truth states
        for expected_states, obs, _ in self.decode_tests:
            found_states = self.generating_hmm.viterbi(obs)["viterbi_states"]
            frac_equal = 1.0 * (expected_states == found_states).sum() / len(expected_states)
            msg = "Failed viterbi test for test case '%s'. Expected at least %s%% accuracy. Got %s%%." % (
                self.name, self.min_frac_equal, frac_equal
            )
            assert_greater_equal(frac_equal, self.min_frac_equal, msg)


class TestACoin(_BaseExample):
    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Coin example"
        cls.min_frac_equal = 0.69

        cls.generating_hmm = FirstOrderHMM(
            **{
                "state_priors": ArrayFactor([0.005, 0.995]),
                "trans_probs": MatrixFactor(numpy.array([[0.8, 0.2], [0.3, 0.7]])),
                "emission_probs": [ArrayFactor([0.6, 0.4]),
                                   ArrayFactor([0.15, 0.85])],
            }
        )

        cls.hmm_dict = {
            "emission_probs": [],
            "state_priors": {
                "shape": (1, 2),
                "row": [0, 0],
                "col": [0, 1],
                "data": [0.005, 0.995],
            },
            "trans_probs": {
                "shape": (2, 2),
                "row": [0, 0, 1, 1],
                "col": [0, 1, 0, 1],
                "data": [0.8, 0.2, 0.3, 0.7],
            }
        }


class TestBTwoGaussian(_BaseExample):
    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Two gaussian example"
        cls.min_frac_equal = 0.7
        transitions = numpy.matrix([[0.9, 0.1], [0.25, 0.75]])

        cls.generating_hmm = FirstOrderHMM(
            **{
                "trans_probs":
                MatrixFactor(transitions),
                "state_priors":
                ArrayFactor([0.8, 0.2]),
                "emission_probs": [
                    ScipyDistributionFactor(scipy.stats.norm, loc=0, scale=0.5),
                    ScipyDistributionFactor(scipy.stats.norm, loc=5, scale=10)
                ],
            }
        )

        cls.hmm_dict = {
            "emission_probs": [],
            "state_priors": {
                "shape": (1, 2),
                "row": [0, 0],
                "col": [0, 1],
                "data": [0.8, 0.2],
            },
            "trans_probs": {
                "shape": (2, 2),
                "row": [0, 0, 1, 1],
                "col": [0, 1, 0, 1],
                "data": [0.9, 0.1, 0.25, 0.75],
            }
        }


@attr("slow")
class TestCFourPoisson(_BaseExample):

    @classmethod
    def do_subclass_setup(cls):
        cls.name = "Four-state Poisson example"
        cls.min_frac_equal = 0.7
        transitions = numpy.matrix(
            [
                [0.8, 0.05, 0.05, 0.1],
                [0.2, 0.6, 0.1, 0.1],
                [0.01, 0.97, 0.01, 0.01],
                [0.45, 0.01, 0.04, 0.5],
            ]
        )

        cls.generating_hmm = get_fourstate_poisson()
        cls.hmm_dict = {
            "emission_probs": [],
            "state_priors": {
                "shape": (1, 4),
                "row": [0, 0, 0, 0],
                "col": [0, 1, 2, 3],
                "data": [0.7, 0.05, 0.15, 0.10],
            },
            "trans_probs": {
                "shape": (4, 4),
                "row": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "col": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "data": [
                    0.8, 0.05, 0.05, 0.1, 0.2, 0.6, 0.1, 0.1, 0.01, 0.97, 0.01, 0.01, 0.45, 0.01,
                    0.04, 0.5
                ],
            }
        }
