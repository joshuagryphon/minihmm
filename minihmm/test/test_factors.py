#!/usr/bin/env python
"""
"""
import pickle
import unittest


import numpy
import jsonpickle
import jsonpickle.ext.numpy
jsonpickle.ext.numpy.register_handlers()

from nose.tools import assert_equal
from minihmm.test.common import (check_equal, check_array_equal, check_tuple_equal)

from minihmm.factors import (
    ArrayFactor,
    MatrixFactor,
    FunctionFactor,
    LogFunctionFactor,
    ScipyDistributionFactor,
)


class AbstractFactor():

    # list of factors
    factors = []

    # list-of-lists of tuples in each factor's domain
    examples = []

    random_seed = 32853294
    to_generate = 50

    @classmethod
    def setUpClass(cls):
        if hasattr(cls, "do_subclass_setup"):
            cls.do_subclass_setup()

        cls.from_json = []
        cls.from_pickle = []
        cls.generated = []

        for factor in cls.factors:
            cls.from_json.append(jsonpickle.decode(jsonpickle.encode(factor)))
            cls.from_pickle.append(pickle.loads(pickle.dumps(factor)))
            numpy.random.seed(seed=cls.random_seed)
            cls.generated.append([factor.generate() for _ in range(cls.to_generate)])

    # override in subclass
    def test_generate_distribution_is_correct(self):
        assert False

    # override in subclass
    def test_logprob_is_correct(self):
        assert False

    # override in subclass
    def test_probability_is_correct(self):
        assert False

    def test_revive_from_pickle_equal(self):
        # NOTE: only tested if class implements __eq__
        for initial, revived in zip(self.factors, self.from_pickle):
            if hasattr(initial, "__eq__"):
                yield check_equal, initial, revived

    def test_revive_from_json_equal(self):
        # NOTE: only tested if class implements __eq__
        for initial, revived in zip(self.factors, self.from_json):
            if hasattr(initial, "__eq__"):
                yield check_equal, initial, revived

    def test_revive_from_pickle_logprob(self):
        for initial, revived, examples in zip(self.factors, self.from_pickle, self.examples):
            for my_example in examples:
                expected = initial.logprob(*my_example)
                found = revived.logprob(*my_example)
                yield check_equal, expected, found

    def test_revive_from_pickle_prob(self):
        for initial, revived, examples in zip(self.factors, self.from_pickle, self.examples):
            for my_example in examples:
                expected = initial.probability(*my_example)
                found = revived.probability(*my_example)
                yield check_equal, expected, found

    def test_revive_from_json_logprob(self):
        for initial, revived, examples in zip(self.factors, self.from_json, self.examples):
            for my_example in examples:
                expected = initial.logprob(*my_example)
                found = revived.logprob(*my_example)
                yield check_equal, expected, found

    def test_revive_from_json_prob(self):
        for initial, revived, examples in zip(self.factors, self.from_json, self.examples):
            for my_example in examples:
                expected = initial.probability(*my_example)
                found = revived.probability(*my_example)
                yield check_equal, expected, found

    def test_revive_from_pickle_generate(self):
        for initial, revived, generated in zip(self.factors, self.from_pickle, self.generated):
            numpy.random.seed(seed=self.random_seed)
            found = [revived.generate() for _ in range(self.to_generate)]
            yield check_array_equal, generated, found

    def test_revive_from_json_generate(self):
        for initial, revived, generated in zip(self.factors, self.from_json, self.generated):
            numpy.random.seed(seed=self.random_seed)
            found = [revived.generate() for _ in range(self.to_generate)]
            yield check_array_equal, generated, found


#===============================================================================
# Tests for ArrayFactor, MatrixFactor
#===============================================================================


class TestArrayFactor(AbstractFactor):
    @classmethod
    def do_subclass_setup(cls):
        for my_len in range(10, 100, 200):
            ary = numpy.random.random(my_len)
            ary /= ary.sum()
            cls.factors.append(ArrayFactor(ary))
            cls.examples.append([(X, ) for X in numpy.random.randint(0, high=my_len, size=50)])

    def test_revive_from_json_shape(self):
        for initial, revived in zip(self.factors, self.from_json):
            yield check_tuple_equal, initial.data.shape, revived.data.shape

    def test_revive_from_pickle_shape(self):
        for initial, revived in zip(self.factors, self.from_pickle):
            yield check_tuple_equal, initial.data.shape, revived.data.shape

    @unittest.skip
    def test_generate_distribution_is_correct(self):
        assert False

    @unittest.skip
    def test_logprob_is_correct(self):
        assert False

    @unittest.skip
    def test_probability_is_correct(self):
        assert False


class TestMatrixFactor(AbstractFactor):
    @classmethod
    def do_subclass_setup(cls):
        for my_len in range(10, 100, 200):
            ary = numpy.random.random((my_len, my_len))
            ary = (ary.T / ary.sum(1)).T
            cls.factors.append(MatrixFactor(ary, row_conditional=True))
            cls.examples.append(
                [(X, Y) for (X, Y) in numpy.random.randint(0, high=my_len, size=(50, 2))]
            )

            cls.factors.append(MatrixFactor(ary, row_conditional=False))
            cls.examples.append(
                [(X, Y) for (X, Y) in numpy.random.randint(0, high=my_len, size=(50, 2))]
            )

    def test_revive_from_json_shape(self):
        for initial, revived in zip(self.factors, self.from_json):
            yield check_tuple_equal, initial.data.shape, revived.data.shape

    def test_revive_from_pickle_shape(self):
        for initial, revived in zip(self.factors, self.from_pickle):
            yield check_tuple_equal, initial.data.shape, revived.data.shape

    @unittest.skip
    def test_generate_distribution_is_correct(self):
        assert False

    @unittest.skip
    def test_logprob_is_correct(self):
        assert False

    @unittest.skip
    def test_probability_is_correct(self):
        assert False


#===============================================================================
# Tests for ScipyDistributionFactor
#===============================================================================

from scipy.stats.distributions import binom, chi2, norm


class TestScipyDistributionFactor(AbstractFactor):

    factors = [
        ScipyDistributionFactor(binom, n=5, p=0.15),
        ScipyDistributionFactor(chi2, df=3),
        ScipyDistributionFactor(norm, loc=0.5, scale=2),
    ]

    examples = [
        [(X, ) for X in numpy.arange(6)],
        [(X, ) for X in numpy.linspace(0, 20, 100)],
        [(X, ) for X in numpy.linspace(-5, 5, 30)],
    ]

    @unittest.skip
    def test_generate_distribution_is_correct(self):
        assert False

    @unittest.skip
    def test_logprob_is_correct(self):
        assert False

    @unittest.skip
    def test_probability_is_correct(self):
        assert False


#===============================================================================
# Tests for FunctionFactor, LogFunctionFactor
#===============================================================================


def _fact_func1(x):
    return 1.0 / numpy.exp(-x)


def _fact_func2(x, **kwargs):
    return x + sum(kwargs.values())


def _gen_func(*args, **kwargs):
    return 5


class TestFunctionFactor(AbstractFactor):

    factors = [
        FunctionFactor(_fact_func1, _gen_func),
        FunctionFactor(_fact_func2, _gen_func, a=5, b=2)
    ]

    examples = [
        [(X, ) for X in numpy.linspace(1, 100, 10)],
        [(X, ) for X in numpy.linspace(1, 100, 10)],
    ]

    @unittest.skip
    def test_generate_distribution_is_correct(self):
        assert False

    @unittest.skip
    def test_logprob_is_correct(self):
        assert False

    @unittest.skip
    def test_probability_is_correct(self):
        assert False


class TestLogFunctionFactor(AbstractFactor):
    factors = [
        LogFunctionFactor(_fact_func1, _gen_func),
        LogFunctionFactor(_fact_func2, _gen_func, a=5, b=2)
    ]

    examples = [
        [(X, ) for X in numpy.linspace(0, 1, 10)],
        [(X, ) for X in numpy.linspace(0, 1, 10)],
    ]

    @unittest.skip
    def test_generate_distribution_is_correct(self):
        assert False

    @unittest.skip
    def test_logprob_is_correct(self):
        assert False

    @unittest.skip
    def test_probability_is_correct(self):
        assert False


