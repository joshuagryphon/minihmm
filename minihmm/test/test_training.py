#!/usr/bin/env vpython
import numpy
import functools

from numpy.testing import (
    assert_almost_equal,
    assert_array_less,
)
from nose.tools import assert_greater_equal
from minihmm.test.common import check_almost_equal

from minihmm.hmm import FirstOrderHMM

from minihmm.estimators import (
    UnivariateGaussianEmissionEstimator,
    DiscreteEmissionEstimator,
    DiscreteStatePriorEstimator,
    DiscreteTransitionEstimator,
)
from minihmm.training import (
    train_baum_welch,
    neg_exp_noise_gen
)
from minihmm.factors import (
    ArrayFactor,
    MatrixFactor,
    LogFunctionFactor,
    ScipyDistributionFactor
)


class _BaseExample:

    @classmethod
    def setUpClass(cls):
        cls.do_subclass_setup()
        print("Setting up tests for %s" % cls.test_name)
        cls.generating_hmm = FirstOrderHMM(state_priors   = ArrayFactor(cls.arrays["state_priors"]),
                                           trans_probs    = MatrixFactor(cls.arrays["transition_probs"]),
                                           emission_probs = cls.get_emission_probs())

        cls.naive_hmm = FirstOrderHMM(state_priors   = ArrayFactor(cls.arrays["naive_state_priors"]),
                                      trans_probs    = MatrixFactor(cls.arrays["naive_transition_probs"]),
                                      emission_probs = cls.get_naive_emission_probs())



        # generate 100 training examples
        cls.training_examples = [cls.generating_hmm.generate(200)[1] for _ in range(100)]

        # retrain
        cls.training_results = train_baum_welch(cls.naive_hmm,
                                                cls.training_examples,
                                                state_prior_estimator = cls.state_prior_estimator,
                                                transition_estimator  = cls.transition_estimator,
                                                emission_estimator    = cls.emission_estimator,
                                                miniter               = 100,
                                                maxiter               = 1000,
                                                processes             = 1
                                               )

    @classmethod
    def do_subclass_setup(cls):
        # must define the following in subclasses
        cls.test_name = None
        cls.arrays            = {}
        cls.state_prior_estimator = None
        cls.transition_estimator  = None
        cls.emission_estimator    = None

    def test_state_priors_trained(self):
        found    = self.training_results["best_model"].state_priors.data
        expected = self.generating_hmm.state_priors.data
        sse = (found - expected)**2
        assert_array_less(sse, 1e-3)

    def test_transition_probs_trained(self):
        found    = self.training_results["best_model"].trans_probs.data
        expected = self.generating_hmm.trans_probs.data
        sse = (found - expected)**2
        assert_array_less(sse, 1e-3)

    def test_likelihoods_increased(self):
        delta = numpy.convolve([1, -1], self.training_results["weight_logprobs"], mode="valid")
        assert_greater_equal((delta >= 0).sum() / len(delta), 0.9)


class TestCasinoTraining(_BaseExample):
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
        cls.state_prior_estimator = DiscreteStatePriorEstimator()
        cls.transition_estimator  = DiscreteTransitionEstimator()
        cls.emission_estimator    = DiscreteEmissionEstimator(6)

        cls.arrays = {
            "state_priors"       : numpy.array([0.99, 0.01]),
            "naive_state_priors" : numpy.array([0.6, 0.4]),
            "transition_probs"   : numpy.array([[0.95, 0.05],
                                              [0.1,  0.9]]),
            "naive_transition_probs" : numpy.array([[0.5, 0.5],
                                                    [0.5, 0.5]]),
            "emission_probs"   : [
                1.0 * numpy.ones(6) / 6.0,
                numpy.array([0.1,0.1,0.1,0.1,0.1,0.5])
            ],
            "naive_emission_probs"   : [
                1.0 * numpy.ones(6) / 6.0,
                1.0 * numpy.ones(6) / 6.0,
            ],
        }

    def test_emission_probs_trained(self):
        for found, expected in zip(self.training_results["best_model"].emission_probs,
                                   self.generating_hmm.emission_probs):
            sse = (found.data - expected.data)**2
            assert_array_less(sse, 1e-4)

def get_casino(hmm_type=FirstOrderHMM):
    """Construct a two-state HMM over discrete values for testing purposes.
    
    This also implicitly tests ArrayFactor and MatrixFactor
    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    state_priors = ArrayFactor(numpy.array([0.99,0.01]))
    fair   = ArrayFactor(1.0 * numpy.ones(6) / 6.0)
    unfair = ArrayFactor(numpy.array([0.1,0.1,0.1,0.1,0.1,0.5]))
    #unfair = ArrayFactor(numpy.array([0.01,0.01,0.01,0.01,0.01,0.95]))
    emission_probs = [fair,unfair]
    transition_probs = MatrixFactor(numpy.array([[0.95,0.05],
                                                 [0.1,0.9]]))
    casino_hmm = hmm_type(state_priors,
                          emission_probs,
                          transition_probs)
    return casino_hmm
