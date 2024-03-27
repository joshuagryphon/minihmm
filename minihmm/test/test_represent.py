#!/usr/bin/env python
"""Unit tests for reducing model order and for changing model representation
"""
import copy
import itertools
import unittest
import warnings

import numpy

from minihmm.factors import MatrixFactor, ArrayFactor
from minihmm.hmm import FirstOrderHMM
from minihmm.represent import ModelReducer
from minihmm.util import matrix_to_dict

from nose.tools import (
    assert_equal,
    assert_greater,
    assert_true,
    assert_almost_equal,
    assert_dict_equal,
    assert_list_equal,
    assert_tuple_equal,
    assert_raises,
)

from numpy.testing import assert_array_equal

from minihmm.test.common import (
    check_equal,
    check_almost_equal,
    check_not_equal,
    check_array_equal,
    check_dict_equal,
    check_list_equal,
    check_tuple_equal,
    check_raises,
    check_true,
    check_none,
    get_dirty_casino,
    get_fourstate_poisson,
)

#===============================================================================
# Tests for ModelReducer
#===============================================================================


class TestModelReducer():

    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.max_order = 6
        cls.max_states = 7
        for num_states in range(2, cls.max_states):
            for starting_order in range(1, cls.max_order):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ModelReducer(starting_order, num_states)
                    cls.models[(starting_order, num_states)] = model

        cls.sequences = [
            [0, 2, 0, 5, 2, 2, 4, 2, 4, 1],
        ]

        # expected tuples for each seq in cls.sequences
        # when moving in model order K
        cls.expected_tuples = {
            2: [
                [(-1, 0), (0, 2), (2, 0), (0, 5), (5, 2), (2, 2), (2, 4), (4, 2), (2, 4), (4, 1)],
            ],
            3: [
                [
                    (-2, -1, 0),
                    (-1, 0, 2),
                    (0, 2, 0),
                    (2, 0, 5),
                    (0, 5, 2),
                    (5, 2, 2),
                    (2, 2, 4),
                    (2, 4, 2),
                    (4, 2, 4),
                    (2, 4, 1),
                ],
            ],
            4: [
                [
                    (-3, -2, -1, 0),
                    (-2, -1, 0, 2),
                    (-1, 0, 2, 0),
                    (0, 2, 0, 5),
                    (2, 0, 5, 2),
                    (0, 5, 2, 2),
                    (5, 2, 2, 4),
                    (2, 2, 4, 2),
                    (2, 4, 2, 4),
                    (4, 2, 4, 1),
                ],
            ],
            5: [
                [
                    (-4, -3, -2, -1, 0),
                    (-3, -2, -1, 0, 2),
                    (-2, -1, 0, 2, 0),
                    (-1, 0, 2, 0, 5),
                    (0, 2, 0, 5, 2),
                    (2, 0, 5, 2, 2),
                    (0, 5, 2, 2, 4),
                    (5, 2, 2, 4, 2),
                    (2, 2, 4, 2, 4),
                    (2, 4, 2, 4, 1),
                ],
            ],
        }

        # keys are (num_states, model_order)
        cls.expected_high_states_to_low = {
            (4, 2): {
                # created by adding new start state
                (-1, 0): 0,
                (-1, 1): 1,
                (-1, 2): 2,
                (-1, 3): 3,

                # represented
                (0, 0): 4,
                (0, 1): 5,
                (0, 2): 6,
                (0, 3): 7,
                (1, 0): 8,
                (1, 1): 9,
                (1, 2): 10,
                (1, 3): 11,
                (2, 0): 12,
                (2, 1): 13,
                (2, 2): 14,
                (2, 3): 15,
                (3, 0): 16,
                (3, 1): 17,
                (3, 2): 18,
                (3, 3): 19,
            },
            (4, 3): {
                # created from new start states
                (-2, -1, 0): 0,
                (-2, -1, 1): 1,
                (-2, -1, 2): 2,
                (-2, -1, 3): 3,
                (-1, 0, 0): 4,
                (-1, 0, 1): 5,
                (-1, 0, 2): 6,
                (-1, 0, 3): 7,
                (-1, 1, 0): 8,
                (-1, 1, 1): 9,
                (-1, 1, 2): 10,
                (-1, 1, 3): 11,
                (-1, 2, 0): 12,
                (-1, 2, 1): 13,
                (-1, 2, 2): 14,
                (-1, 2, 3): 15,
                (-1, 3, 0): 16,
                (-1, 3, 1): 17,
                (-1, 3, 2): 18,
                (-1, 3, 3): 19,

                # actual states
                (0, 0, 0): 20,
                (0, 0, 1): 21,
                (0, 0, 2): 22,
                (0, 0, 3): 23,
                (0, 1, 0): 24,
                (0, 1, 1): 25,
                (0, 1, 2): 26,
                (0, 1, 3): 27,
                (0, 2, 0): 28,
                (0, 2, 1): 29,
                (0, 2, 2): 30,
                (0, 2, 3): 31,
                (0, 3, 0): 32,
                (0, 3, 1): 33,
                (0, 3, 2): 34,
                (0, 3, 3): 35,
                (1, 0, 0): 36,
                (1, 0, 1): 37,
                (1, 0, 2): 38,
                (1, 0, 3): 39,
                (1, 1, 0): 40,
                (1, 1, 1): 41,
                (1, 1, 2): 42,
                (1, 1, 3): 43,
                (1, 2, 0): 44,
                (1, 2, 1): 45,
                (1, 2, 2): 46,
                (1, 2, 3): 47,
                (1, 3, 0): 48,
                (1, 3, 1): 49,
                (1, 3, 2): 50,
                (1, 3, 3): 51,
                (2, 0, 0): 52,
                (2, 0, 1): 53,
                (2, 0, 2): 54,
                (2, 0, 3): 55,
                (2, 1, 0): 56,
                (2, 1, 1): 57,
                (2, 1, 2): 58,
                (2, 1, 3): 59,
                (2, 2, 0): 60,
                (2, 2, 1): 61,
                (2, 2, 2): 62,
                (2, 2, 3): 63,
                (2, 3, 0): 64,
                (2, 3, 1): 65,
                (2, 3, 2): 66,
                (2, 3, 3): 67,
                (3, 0, 0): 68,
                (3, 0, 1): 69,
                (3, 0, 2): 70,
                (3, 0, 3): 71,
                (3, 1, 0): 72,
                (3, 1, 1): 73,
                (3, 1, 2): 74,
                (3, 1, 3): 75,
                (3, 2, 0): 76,
                (3, 2, 1): 77,
                (3, 2, 2): 78,
                (3, 2, 3): 79,
                (3, 3, 0): 80,
                (3, 3, 1): 81,
                (3, 3, 2): 82,
                (3, 3, 3): 83
            },
        }

        # simple numerical array of observations
        cls.test_array_high = 6
        cls.test_array = numpy.random.randint(0, high=cls.test_array_high, size=10)

    @staticmethod
    def revdict(d):
        # make sure we won't overwrite entries
        vals = list(d.values())
        assert len(set(vals)) == len(vals)
        return {v: k for (k, v) in d.items()}

    @classmethod
    def get_random_hmm(cls, model):
        """Get a randomly-initialized HMM matching the low-order states for
        a :class:`ModelReducer`

        Parameters
        ----------
        model : ModelReducer
        
        Returns
        -------
        :class:`minihmm.hmm.FirstOrderHMM`
            Randomly-initialized first-order HMM congruent with the low-order
            state space of `model`
        """
        state_priors = numpy.random.random(model.low_order_states)
        state_priors /= state_priors.sum()

        trans_probs = numpy.random.random((model.low_order_states, model.low_order_states))
        trans_probs = (trans_probs.T / trans_probs.sum(1)).T

        emissions = []
        for _ in range(model.low_order_states):
            my_array = numpy.random.random(cls.test_array_high)
            my_array /= my_array.sum()
            emissions.append(ArrayFactor(my_array))

        my_hmm = FirstOrderHMM(
            state_priors=ArrayFactor(state_priors),
            trans_probs=MatrixFactor(trans_probs),
            emission_probs=emissions,
        )
        return my_hmm


    # tests start here --------------------------------------------------------

    def test_init_bad_parameters_raises_value_error(self):
        for numstates, model_order in itertools.product([-1, 0, 5], [-2, -1, 0]):
            yield check_raises, ValueError, ModelReducer, model_order, numstates

    def test_init_on_first_order_is_identical(self):
        expected = {(0, ): 0, (1, ): 1, (2, ): 2, (3, ): 3}
        model = ModelReducer(1, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert_dict_equal(model.high_states_to_low, expected)
            assert_dict_equal(model.low_states_to_high, self.revdict(expected))

    def test_eq(self):
        for k1, k2 in itertools.combinations(self.models.keys(), r=2):
            if k1 == k2:
                yield check_equal, self.models[k1], self.models[k2]
            else:
                yield check_not_equal, self.models[k1], self.models[k2]

    def test_to_dict_no_hmm(self):
        for (starting_order, num_states), model in sorted(self.models.items()):
            found = model._to_dict()
            expected = {
                "starting_order": starting_order,
                "num_states": num_states,
            }
            yield check_equal, found, expected

    def test_to_dict_with_hmm(self):
        for (starting_order, num_states), model in sorted(self.models.items()):
            if starting_order >= 5:
                continue

            my_hmm = self.get_random_hmm(model)
            model.hmm = my_hmm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = model._to_dict()

            expected = {
                "starting_order": starting_order,
                "num_states": num_states,
                "hmm": my_hmm,
            }
            model.hmm = None
            for k in expected:
                if k == "first_order_hmm":
                    yield check_dict_equal, found[k], expected[k]
                else:
                    yield check_equal, found[k], expected[k]

    def test_from_dict_no_hmm(self):
        for (starting_order, num_states), model in sorted(self.models.items()):
            dtmp = {
                "starting_order": starting_order,
                "num_states": num_states,
            }
            found = ModelReducer._from_dict(dtmp)
            expected = ModelReducer(starting_order, num_states)

            for k in ("starting_order", "high_order_states"):
                yield check_equal, getattr(found, k), getattr(expected, k)

            yield check_none, found._hmm

    def test_from_dict_with_hmm(self):
        for (starting_order, num_states), model in sorted(self.models.items()):
            if starting_order >= 5:
                continue

            state_priors = numpy.random.random(model.low_order_states)
            state_priors /= state_priors.sum()

            trans_probs = numpy.random.random((model.low_order_states, model.low_order_states))
            trans_probs = (trans_probs.T / trans_probs.sum(1)).T

            my_hmm = FirstOrderHMM(
                state_priors=ArrayFactor(state_priors),
                trans_probs=MatrixFactor(trans_probs),
                emission_probs=[None] * model.low_order_states
            )

            dtmp = {
                "starting_order": starting_order,
                "num_states": num_states,
                "hmm": my_hmm,
            }

            found = ModelReducer._from_dict(dtmp)
            expected = ModelReducer(starting_order, num_states)
            expected.hmm = my_hmm

            for k in ("starting_order", "high_order_states"):
                yield check_equal, getattr(found, k), getattr(expected, k)

            for k in ("trans_probs", "state_priors"):
                yield check_array_equal, getattr(found.hmm, k).data, getattr(my_hmm, k).data

    def test_to_from_json(self):
        for k, v in self.models.items():
            enc = v.to_json()
            dec = ModelReducer.from_json(enc)
            yield check_equal, v, dec

    def test_transcode_sequence(self):
        testseq = ["A", "B", "C", "D", "E"]
        expected = numpy.arange(5)
        dtmp = {K: V for K, V in zip(testseq, expected)}

        found = ModelReducer.transcode_sequence(testseq, dtmp)
        yield check_array_equal, found, expected

    def test_transcode_sequences(self):
        testseqs = [["A", "B", "C", "D", "E"], ["D", "B", "A", "A"]]

        expected = [numpy.arange(5), numpy.array([3, 1, 0, 0])]

        dtmp = {K: V for K, V in zip(testseqs[0], expected[0])}

        found = ModelReducer.transcode_sequences(testseqs, dtmp)
        yield check_equal, len(found), len(expected)
        for my_found, my_expected in zip(found, expected):
            yield check_array_equal, my_found, my_expected

    def test_get_dummy_states(self):
        for num_states in range(2, self.max_states):
            for starting_order in range(1, self.max_order):
                dummies = self.models[(starting_order, num_states)]._dummy_states
                expected = list(reversed([-X for X in range(1, starting_order)]))
                yield check_list_equal, dummies, expected

    # get_state_mapping() is called in __init__ when models are created,
    # and sets model.high_states_to_low as well as model.low_states_to_high
    # We test by inspecting models created in setUpClass
    def test_get_state_mapping_high_states_to_low(self):
        for (num_states, model_order), expected in self.expected_high_states_to_low.items():
            found = self.models[(model_order, 4)].high_states_to_low
            yield check_dict_equal, found, expected

    # get_state_mapping() is called in __init__ when models are created,
    # and sets model.high_states_to_low as well as model.low_states_to_high
    # We test by inspecting models created in setUpClass
    def test_get_state_mapping_low_states_to_high(self):
        for (num_states, model_order), expected in self.expected_high_states_to_low.items():
            found = self.models[(model_order, 4)].low_states_to_high
            yield check_dict_equal, found, self.revdict(expected)

    def check_get_stateseq_tuples(self, model_order):
        # n.b. test assumes _get_dummy_states is working
        model = self.models[(model_order, 6)]

        expected = self.expected_tuples[model_order]
        found = model._get_stateseq_tuples(self.sequences)
        assert_equal(
            len(found),
            len(expected),
            (
                "Number of output sequences '%s' does not match number of "
                "input sequences '%s' for _get_stateseq_tuples(), order '%s'"
                % (len(found), len(expected), model_order)
            )
        )
        for e, f in zip(expected, found):
            assert_list_equal(e, f)

    def test_get_stateseq_tuples_forward(self):
        for model_order in self.expected_tuples:
            yield self.check_get_stateseq_tuples, model_order

    def check_lower_stateseq_orders(self, model_order):
        #requires _get_dummy_states, and get_state_mapping to function
        model = self.models[(model_order, 6)]
        forward = model.high_states_to_low
        dummy_states = model._dummy_states
        expected = []
        for my_seq in self.sequences:

            # prepend dummy states to sequence
            my_seq = sorted(dummy_states) + my_seq

            # translate
            expected.append(
                numpy.array(
                    [
                        forward[tuple(my_seq[X:X + model_order])]
                        for X in range(0,
                                       len(my_seq) - model_order + 1)
                    ]
                )
            )

        found = model.lower_stateseq_orders(self.sequences)
        assert len(found) > 0
        assert_equal(
            len(expected), len(found),
            (
                "Number of output sequences '%s' does not match number of "
                "input sequences '%s' for reduce_stateseq_orders, order '%s'"
                % (len(found), len(self.sequences), model_order)
            )
        )
        for e, f in zip(expected, found):
            print("---------------------------------------")
            print(e)
            print(f)
            assert_array_equal(f, e)

    # NOTE: check function assumes high_states_to_low is correct
    def test_lower_stateseq_orders(self):
        assert_greater(len(self.expected_tuples), 0)
        for model_order in self.expected_tuples:
            yield self.check_lower_stateseq_orders, model_order

    def test_lower_statseq_orders_negative_input_states_raises_value_error(self):
        model = list(self.models.values())[0]
        assert_raises(ValueError, model.lower_stateseq_orders, [[5, 1, 3, 4, 1, 0, -1]])
        assert_raises(ValueError, model.lower_stateseq_orders, [[-5, 1, 3, 4, 1, 0, 1]])
        assert_raises(ValueError, model.lower_stateseq_orders, [[5, 1, 3, -4, 1, 0, 1]])

    def check_raise_stateseq_orders(self, model_order):
        model = self.models[(model_order, 6)]
        expected = self.sequences
        inp = (
            [model.high_states_to_low[X] for X in Y] for Y in self.expected_tuples[model_order]
        )
        found = model.raise_stateseq_orders(inp)
        assert_equal(len(found), len(expected))
        for f, e in zip(found, expected):
            assert_array_equal(f, e)

    # NOTE: check function assumes high_states_to_low is correct
    def test_raise_stateseq_orders(self):
        for model_order in self.expected_tuples:
            yield self.check_raise_stateseq_orders, model_order

    # Gold standard would be to create a high order HMM, generate sequences
    # from it in high order space, save results, create an equivalent low-order
    # HMM, and run the unit tests below
    def test_viterbi(self):
        # FIXME: right now, this test only makes sure the method executes.
        # It does *not* verify correctness of output.
        mod = copy.deepcopy(self.models[(2, 2)])
        my_hmm = self.get_random_hmm(mod)
        mod.hmm = my_hmm
        mod.viterbi(self.test_array)

    def test_posterior_decode(self):
        # FIXME: right now, this test only makes sure the method executes.
        # It does *not* verify correctness of output.
        mod = copy.deepcopy(self.models[(2, 2)])
        my_hmm = self.get_random_hmm(mod)
        mod.hmm = my_hmm
        mod.posterior_decode(self.test_array)

    def test_sample(self):
        # FIXME: right now, this test only makes sure the method executes.
        # It does *not* verify correctness of output.
        mod = copy.deepcopy(self.models[(2, 2)])
        my_hmm = self.get_random_hmm(mod)
        mod.hmm = my_hmm
        mod.sample(self.test_array, 2)

    def test_generate(self):
        # FIXME: right now, this test only makes sure the method executes.
        # It does *not* verify correctness of output.
        #
        # Gold standard: sample 10k times, and make sure parameters converge
        # onto high-order transitions and emissions
        mod = copy.deepcopy(self.models[(2, 2)])
        my_hmm = self.get_random_hmm(mod)
        mod.hmm = my_hmm
        mod.generate(200)

    def test_joint_path_logprob(self):
        # FIXME: right now, this test only makes sure the method executes.
        # It does *not* verify correctness of output.
        mod = copy.deepcopy(self.models[(2, 2)])
        my_hmm = self.get_random_hmm(mod)
        mod.hmm = my_hmm

        my_path = numpy.random.randint(0, high=2, size=len(self.test_array))
        mod.joint_path_logprob(my_path, self.test_array)
 
    def test_remap_emission_factors(self):
        for num_states in range(2, self.max_states):
            for starting_order in range(1, self.max_order):
                model = self.models[(starting_order, num_states)]
                factors = list("abcdefg"[:num_states])

                # expected length is number starting states times the number of
                # paths to each, including paths via dummy states
                elen = (num_states**numpy.arange(1, starting_order + 1)).sum()
                mult = elen // num_states

                f_expected = factors * mult
                f_found = model.remap_emission_factors(factors)
                yield check_list_equal, f_found, f_expected

    def test_get_pseudocount_arrays_transitions(self):
        for num_states in range(2, self.max_states):
            for starting_order in range(1, self.max_order):
                model = self.models[(starting_order, num_states)]
                _, pmat = model.get_pseudocount_arrays()

                for (x, y, v) in zip(pmat.row, pmat.col, pmat.data):
                    from_state = model.low_states_to_high[x]
                    to_state = model.low_states_to_high[y]

                    yield check_tuple_equal, from_state[1:], to_state[:-1]

    def test_get_pseudocount_arrays_state_priors(self):
        for num_states in range(2, self.max_states):
            for starting_order in range(1, self.max_order):
                model = self.models[(starting_order, num_states)]
                dummies = model._dummy_states
                found, _ = model.get_pseudocount_arrays()
                expected = numpy.zeros(model.low_order_states)
                for i in range(model.high_order_states):
                    k = tuple(dummies + [i])
                    v = model.high_states_to_low[k]
                    expected[v] = 1

                yield check_array_equal, found, expected

    def test_get_emission_mapping(self):
        cases = {
            (2, 2): numpy.tile(range(2), 3),
            (2, 3): numpy.tile(range(3), 4),
            (2, 4): numpy.tile(range(4), 5),
            (3, 2): numpy.tile(range(2), 7),
            (3, 3): numpy.tile(range(3), 13),
            (3, 4): numpy.tile(range(4), 21),
            (4, 2): numpy.tile(range(2), 15),
            (4, 3): numpy.tile(range(3), 40),
            (4, 4): numpy.tile(range(4), 85),
        }
        for (order, states), expected in sorted(cases.items()):
            found = ModelReducer(order, states).get_emission_mapping()
            yield check_array_equal, expected, found

    def test_remap_from_first_order(self):
        # test parameter remapping by asserting that log probabilities of
        # observation sequences from remapped high-order mdodels match those of
        # low order models

        models = {
            2: get_dirty_casino(),
            4: get_fourstate_poisson(),
        }

        for num_states, native_model in sorted(models.items()):
            seqs = [native_model.generate(200)[1] for _ in range(5)]
            for starting_order in 2, 3, 4, 5:
                mod = ModelReducer(starting_order, num_states)
                trans_model = mod.remap_from_first_order(native_model)

                for my_obs in seqs:
                    native_logprob = native_model.logprob(my_obs)
                    trans_logprob = native_model.logprob(my_obs)

                    yield check_almost_equal, native_logprob, trans_logprob
