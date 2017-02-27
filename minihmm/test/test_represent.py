#!/usr/bin/env python
"""Unit tests for reducing model order and for changing model representation
"""
import unittest
import warnings
import itertools
import numpy

from minihmm.represent import (
    get_state_mapping,
    _get_dummy_states,
    _get_stateseq_tuples,
    raise_stateseq_orders,
    lower_stateseq_orders,
    transcode_sequences,
    remap_emission_factors,
    #reduce_model_order,
    )

from nose.tools import (
    assert_equal,
    assert_greater,
    assert_true,
    assert_almost_equal,
    assert_dict_equal,
    assert_list_equal,
    assert_raises
    )

from numpy.testing import assert_array_equal



class TestGetDummyStates():

    @staticmethod
    def check_results(starting_order, found):
        expected = list(reversed([-X for X in range(1, starting_order)]))
        assert_list_equal(expected, found)

    def test_get_dummy_states(self):
        for starting_order in range(4):
            dummies = _get_dummy_states(starting_order)
            yield self.check_results, starting_order, dummies


class TestLowerParameterOrder():

    @classmethod
    def setUpClass(cls):
        pass

    def test_remap_emission_factors(self):
        for num_states in (2, 3, 4):
            for starting_order in (1, 2, 3, 4):
                factors    = list("abcdefg"[:num_states])

                # expected length is number starting states times the number of paths to each, including
                # paths via dummy states
                elen = (num_states ** numpy.arange(1, starting_order+1)).sum()
                mult = elen / num_states

                f_expected = factors * mult
                f_found    = remap_emission_factors(num_states, factors, starting_order=starting_order)
                assert_list_equal(f_expected, f_found)

    @unittest.skip
    def test_remap_state_priors(self):
        assert False

    @unittest.skip
    def test_remap_transitions(self):
        assert False


class TestLowerStateSequenceManipulation():
    """Tests lower_stateseq_orders, and implicitly tests _get_stateseq_tuples and transcode_sequences)"""
 
    @classmethod
    def setUpClass(cls):
        cls.num_states = 7
 
        cls.sequences = [
            [0, 2, 0, 5, 2, 2, 4, 2, 4, 1],
        ]
 
        cls.expected_tuples = {
            2 : [[(-1,0),
                  ( 0,2),
                  ( 2,0),
                  ( 0,5),
                  ( 5,2),
                  ( 2,2),
                  ( 2,4),
                  ( 4,2),
                  ( 2,4),
                  ( 4,1)],
            ],
 
            3 : [[(-2,-1, 0),
                  (-1, 0, 2),
                  ( 0, 2, 0),
                  ( 2, 0, 5),
                  ( 0, 5, 2),
                  ( 5, 2, 2),
                  ( 2, 2, 4),
                  ( 2, 4, 2),
                  ( 4, 2, 4),
                  ( 2, 4, 1),
                  ],
 
            ],
 
            4 : [[(-3,-2,-1, 0),
                  (-2,-1, 0, 2),
                  (-1, 0, 2, 0),
                  ( 0, 2, 0, 5),
                  ( 2, 0, 5, 2),
                  ( 0, 5, 2, 2),
                  ( 5, 2, 2, 4),
                  ( 2, 2, 4, 2),
                  ( 2, 4, 2, 4),
                  ( 4, 2, 4, 1),
                  ],
            ],
 
            5 : [[(-4, -3,-2,-1, 0),
                  (-3, -2,-1, 0, 2),
                  (-2, -1, 0, 2, 0),
                  (-1,  0, 2, 0, 5),
                  ( 0, 2, 0, 5, 2),
                  ( 2, 0, 5, 2, 2),
                  ( 0, 5, 2, 2, 4),
                  ( 5, 2, 2, 4, 2),
                  ( 2, 2, 4, 2, 4),
                  ( 2, 4, 2, 4, 1),
                  ],
            ],
        }
        
        cls.num_states   = 6
 
    def check_get_stateseq_tuples(self, model_order):
        # n.b. test assumes _get_dummy_states is working
        dummy_states = _get_dummy_states(model_order)
        expected = self.expected_tuples[model_order]
        found    = _get_stateseq_tuples(self.sequences, dummy_states, starting_order=model_order)
        assert_equal(len(expected),len(found),
                "Number of output sequences '%s' does not match number of input sequences '%s' for _get_stateseq_tuples(), order '%s'" % (len(found),len(expected),model_order))
        for e, f in zip(expected,found):
            assert_list_equal(e,f)

 
    def check_reduced(self, model_order):
        #requires _get_dummy_states, and get_state_mapping to function
        dummy_states = _get_dummy_states(model_order)
        forward, _ = get_state_mapping(self.num_states, dummy_states, starting_order=model_order)
        expected   = []
        for my_seq in self.sequences:

            # prepend dummy states to sequence
            my_seq = sorted(dummy_states) + my_seq

            # translate
            expected.append(numpy.array([forward[tuple(my_seq[X:X+model_order])] for X in range(0,len(my_seq)-model_order + 1)])) 
        
        dtmp  = lower_stateseq_orders(self.sequences, self.num_states, starting_order=model_order)
        found = dtmp["state_seqs"]
        assert len(found) > 0
        assert_equal(len(expected),len(found),
                "Number of output sequences '%s' does not match number of input sequences '%s' for reduce_stateseq_orders, order '%s'" % (len(found),len(self.sequences),model_order))
        for e,f in zip(expected, found):
            print("---------------------------------------")
            print(e)
            print(f)
            assert_array_equal(e, f)
 
    def test_get_stateseq_tuples_forward(self):
        for model_order in self.expected_tuples:
            yield self.check_get_stateseq_tuples, model_order
 
    def test_raise_stateseq_orders(self):
        assert False

    def test_lower_stateseq_orders(self):
        assert_greater(len(self.expected_tuples),0)
        for model_order in self.expected_tuples:
            yield self.check_reduced, model_order
 
    def test_negative_input_states_raises_value_error(self):
        assert_raises(ValueError,lower_stateseq_orders,[[5,1,3,4,1,0,-1]],self.num_states,3)
        assert_raises(ValueError,lower_stateseq_orders,[[-5,1,3,4,1,0,1]],self.num_states,3)
        assert_raises(ValueError,lower_stateseq_orders,[[5,1,3,-4,1,0,1]],self.num_states,3)

    
class TestGetStateMapping():
 
    # keys are (num_states, model_order)
    expected_forward = {
                        
        (4, 2) :{
            # created by adding new start state
            (-1, 0): 0,
            (-1, 1): 1,
            (-1, 2): 2,
            (-1, 3): 3,
 
            # represented
            (0, 0) : 4,
            (0, 1) : 5,
            (0, 2) : 6,
            (0, 3) : 7,
            (1, 0) : 8,
            (1, 1) : 9,
            (1, 2) : 10,
            (1, 3) : 11,
            (2, 0) : 12,
            (2, 1) : 13,
            (2, 2) : 14,
            (2, 3) : 15,
            (3, 0) : 16,
            (3, 1) : 17,
            (3, 2) : 18,
            (3, 3) : 19,
        },
                        
        (4, 3) : {
            # created from new start states
            (-2, -1, 0): 0,
            (-2, -1, 1): 1,
            (-2, -1, 2): 2,
            (-2, -1, 3): 3,
            (-1,  0, 0): 4,
            (-1,  0, 1): 5,
            (-1,  0, 2): 6,
            (-1,  0, 3): 7,
            (-1,  1, 0): 8,
            (-1,  1, 1): 9,
            
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
 
    @staticmethod
    def revdict(d):
        # make sure we won't overwrite entries
        vals = d.values()
        assert len(set(vals)) == len(vals)
        return { v : k for (k,v) in d.items() }
 
    @staticmethod
    def check_results(states, model_order, expected_htf):
        """Helper function to check forward and reverse dicts coming from :func:`get_state_mapping`
 
        Parameters
        ----------
        states : int
            Number of states in high-order model
 
        model_order : int
            Order of high-order model
 
        expected_htf : dict
            Dictionary mapping `model-order`-tuples of high-order states to
            their new first-order representations
 
        name : str
            Name of test
        """
        dummy_states = list(reversed([-X for X in range(1, model_order)]))
        found_htf, found_fth = get_state_mapping(states, dummy_states, model_order)
        expected_fth = TestGetStateMapping.revdict(expected_htf)
 
        assert_dict_equal(expected_htf,found_htf)
        assert_dict_equal(expected_fth,found_fth)

    def test_orders(self):
        for (num_states, model_order), expected_htf in self.expected_forward.items():
            yield self.check_results, num_states, model_order, expected_htf
             
    #get_state_mapping(num_states, dummy_states, starting_order=2)
    def test_bad_parameters_raises_value_error(self):
        for numstates, model_order in itertools.product([-1,0,5],[-2,-1,0]):
            assert_raises(ValueError, get_state_mapping, numstates, [], model_order)

    def test_first_order_is_identical(self):
        expected = {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning) 
            yield self.check_results, 4, 1, expected


def test_transcode_sequences():
    testseq  = ["A","B","C","D","E"]
    expected = numpy.arange(5)
    dtmp = {K : V for K,V in zip(testseq,expected)}
    
    found = transcode_sequences([testseq], dtmp)
    assert_array_equal(expected, found[0])
