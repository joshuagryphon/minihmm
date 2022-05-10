#!/usr/bin/env vpython
import sys
import unittest

import numpy
import scipy.stats

from numpy.testing import (
    assert_almost_equal,
)
from nose.plugins.attrib import attr
from nose.tools import assert_greater_equal, assert_true
from minihmm.test.common import check_almost_equal

from minihmm.hmm import FirstOrderHMM

from minihmm.estimators import (
    UnivariateGaussianEmissionEstimator,
    DiscreteEmissionEstimator,
    DiscreteStatePriorEstimator,
    DiscreteTransitionEstimator,
)
from minihmm.training import (train_baum_welch, neg_exp_noise_gen, DefaultLoggerFactory)
from minihmm.factors import (ArrayFactor, MatrixFactor, ScipyDistributionFactor)


class ScreenWriter:
    @staticmethod
    def write(inp):
        sys.stdout.write(inp + "\n")


class _BaseExample:

    @classmethod
    def setUpClass(cls):
        cls.do_subclass_setup()
        print("Setting up tests for %s" % cls.test_name)
        cls.generating_hmm = FirstOrderHMM(
            state_priors=ArrayFactor(cls.arrays["state_priors"]),
            trans_probs=MatrixFactor(cls.arrays["transition_probs"]),
            emission_probs=cls.get_emission_probs()
        )

        cls.naive_hmm = FirstOrderHMM(
            state_priors=ArrayFactor(cls.arrays["naive_state_priors"]),
            trans_probs=MatrixFactor(cls.arrays["naive_transition_probs"]),
            emission_probs=cls.get_naive_emission_probs()
        )

        # generate 100 training examples
        cls.training_examples = [
            cls.generating_hmm.generate(cls.example_len)[1] for _ in range(cls.num_examples)
        ]

        # retrain
        cls.training_results = train_baum_welch(
            cls.naive_hmm,
            cls.training_examples,
            state_prior_estimator=cls.state_prior_estimator,
            transition_estimator=cls.transition_estimator,
            emission_estimator=cls.emission_estimator,
            miniter=80,
            maxiter=1000,
            processes=1,
            logfunc=DefaultLoggerFactory(ScreenWriter(), cls.naive_hmm, maxcols=5),
        )

    @classmethod
    def do_subclass_setup(cls):
        # must define the following in subclasses
        cls.test_name = None
        cls.arrays = {}
        cls.example_len = 200
        cls.num_examples = 100
        cls.state_prior_estimator = None
        cls.transition_estimator = None
        cls.emission_estimator = None

    def test_state_priors_trained(self):
        found = self.training_results["best_model"].state_priors.data
        expected = self.generating_hmm.state_priors.data
        yield check_almost_equal, found, expected, {"decimal": 1}

    def test_transition_probs_trained(self):
        found = self.training_results["best_model"].trans_probs.data
        expected = self.generating_hmm.trans_probs.data
        yield check_almost_equal, found, expected, {"decimal": 2}

    def test_likelihoods_increased_each_round(self):
        delta = numpy.convolve([1, -1], self.training_results["weight_logprobs"], mode="valid")
        assert_greater_equal((delta >= 0).sum() / len(delta), 0.9)

    # override this method in subclass
    def test_emission_probs_trained(self):
        assert False


class TestCasino(_BaseExample):
    """Test re-training on a two-state HMM with discrete emissions"""

    @classmethod
    def get_emission_probs(cls):
        return [ArrayFactor(X) for X in cls.arrays["emission_probs"]]

    @classmethod
    def get_naive_emission_probs(cls):
        return [ArrayFactor(X) for X in cls.arrays["naive_emission_probs"]]

    @classmethod
    def do_subclass_setup(cls):
        cls.test_name = "DiscreteCasino"
        cls.example_len = 200
        cls.num_examples = 150
        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator = DiscreteTransitionEstimator()
        cls.emission_estimator = DiscreteEmissionEstimator(6)

        cls.arrays = {
            "state_priors": numpy.array([0.99, 0.01]),
            "naive_state_priors": numpy.array([0.6, 0.4]),
            "transition_probs": numpy.array([[0.95, 0.05], [0.1, 0.9]]),
            "naive_transition_probs": numpy.array([[0.6, 0.4], [0.5, 0.5]]),
            "emission_probs":[
                1.0 * numpy.ones(6) / 6.0,
                numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            ],
            "naive_emission_probs": [
                1.0 * numpy.ones(6) / 6.0,
                1.0 * numpy.ones(6) / 6.0,
            ],
        }

    def test_emission_probs_trained(self):
        for found, expected in zip(
               self.training_results["best_model"].emission_probs,
               self.generating_hmm.emission_probs
            ):
            #sse = (found.data - expected.data)**2
            #assert_true((sse < 1e-4).all())
            yield check_almost_equal, found.data, expected.data, {"decimal": 2}



@attr("slow")
class TestGaussian(_BaseExample):
    """Test re-training on a three-state HMM with continuous emissions"""

    @classmethod
    def get_emission_probs(cls):
        return [
            ScipyDistributionFactor(scipy.stats.norm, loc=0, scale=0.5),
            ScipyDistributionFactor(scipy.stats.norm, loc=5, scale=3),
            ScipyDistributionFactor(scipy.stats.norm, loc=-2, scale=1),
        ]

    @classmethod
    def get_naive_emission_probs(cls):
        return [
            ScipyDistributionFactor(scipy.stats.norm, loc=0, scale=1),
            ScipyDistributionFactor(scipy.stats.norm, loc=0, scale=1),
            ScipyDistributionFactor(scipy.stats.norm, loc=0, scale=1),
        ]

    @classmethod
    def do_subclass_setup(cls):
        cls.test_name = "ContinuousGaussian"
        cls.example_len = 500
        cls.num_examples = 150
        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator = DiscreteTransitionEstimator()
        cls.emission_estimator = UnivariateGaussianEmissionEstimator()

        cls.arrays = {
            "state_priors": numpy.array([0.8, 0.05, 0.15]),
            "naive_state_priors": numpy.array([0.6, 0.2, 0.2]),
            "transition_probs": numpy.array([
                [0.9, 0.05, 0.05],
                [0.1, 0.6, 0.3],
                [0.2, 0.1, 0.7],
            ]),
            "naive_transition_probs": numpy.array([
                [0.40, 0.30, 0.30],
                [0.30, 0.36, 0.33],
                [0.33, 0.30, 0.36],
            ]),
        }

    def test_emission_probs_trained(self):
        for found, expected in zip(
            self.training_results["best_model"].emission_probs,
            self.generating_hmm.emission_probs
        ):
            for k in expected.dist_kwargs.keys():
                yield (
                    check_almost_equal,
                    found.dist_kwargs[k],
                    expected.dist_kwargs[k],
                    { "decimal": 2 },
                )


