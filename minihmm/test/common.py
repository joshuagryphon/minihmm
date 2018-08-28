#!/usr/bin/env python
import numpy
from minihmm import *
from minihmm.factors import *
from minihmm.represent import *

from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
)

from nose.tools import (
    assert_equal,
    assert_true,
    assert_dict_equal,
    assert_list_equal,
    assert_tuple_equal,
    assert_raises,
)

#===============================================================================
# Covenience functions to deal with test generators
#===============================================================================


def check_equal(a, b, msg=None):
    assert_equal(a, b, msg)


def check_array_equal(a, b, **kwargs):
    assert_array_equal(a, b, **kwargs)


def check_almost_equal(a, b, kwargs={}):
    assert_almost_equal(a, b, **kwargs)


def check_true(a, kwargs={}):
    assert_true(a, **kwargs)


def check_none(a, msg=None):
    assert_true(a is None, msg=msg)


def check_not_equal(a, b):
    assert_raises(AssertionError, check_array_equal, a, b)


def check_list_equal(a, b, msg=None):
    assert_list_equal(a, b, msg)


def check_dict_equal(a, b, msg=None):
    assert_dict_equal(a, b, msg)


def check_tuple_equal(a, b, msg=None):
    assert_tuple_equal(a, b, msg)


def check_raises(cls, callable_, *args):
    assert_raises(cls, callable_, *args)


#===============================================================================
# Pre=built HMMs for testing and examples
#===============================================================================


def get_dirty_casino():
    """Return a two-state HMM similar to the "dirty" casino example from Durbin et al."""
    state_priors = ArrayFactor(numpy.array([0.9, 0.1]))
    trans_probs = MatrixFactor(numpy.array([[0.85, 0.15], [0.4, 0.6]]))
    emission_probs = [
        ArrayFactor(numpy.tile((1.0 / 6), 6)),
        ArrayFactor(numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]))
    ]
    return FirstOrderHMM(
        state_priors=state_priors, trans_probs=trans_probs, emission_probs=emission_probs
    )


def get_fourstate_poisson():
    """Return a four-state HMM with Poisson-distributed emission for each state"""
    transitions = numpy.array(
        [
            [0.8, 0.05, 0.05, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.01, 0.97, 0.01, 0.01],
            [0.45, 0.01, 0.04, 0.5],
        ]
    )
    trans_probs = MatrixFactor(transitions)
    state_priors = ArrayFactor([0.7, 0.05, 0.15, 0.10])
    emission_probs = [
        ScipyDistributionFactor(scipy.stats.poisson, 1),
        ScipyDistributionFactor(scipy.stats.poisson, 5),
        ScipyDistributionFactor(scipy.stats.poisson, 10),
        ScipyDistributionFactor(scipy.stats.poisson, 25),
    ]

    return FirstOrderHMM(
        state_priors=state_priors, trans_probs=trans_probs, emission_probs=emission_probs
    )
