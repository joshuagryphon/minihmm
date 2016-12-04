#!/usr/bin/env python
"""
"""

import numpy

from minihmm.represent import (
    transcode_sequences,
    get_state_mapping,
    #reduce_model_order,
    )

from nose.tools import assert_equal, assert_true, assert_almost_equal, assert_dict_equal


class TestGetStateMapping():

    @staticmethod
    def revdict(d):
        # make sure we won't overwrite entries
        vals = d.values()
        assert len(set(vals)) == len(vals)
        return { v : k for (k,v) in d.items() }

    def test_second_order(self):
        states = 4
        expected_high_to_first = {
            (-2, -2): 0,
            (-2, -1): 1,
            (-2, 0): 2,
            (-2, 1): 3,
            (-2, 2): 4,
            (-2, 3): 5,
            (-1, -2): 6,
            (-1, -1): 7,
            (-1, 0): 8,
            (-1, 1): 9,
            (-1, 2): 10,
            (-1, 3): 11,
            (0, -2): 12,
            (0, -1): 13,
            (0, 0): 14,
            (0, 1): 15,
            (0, 2): 16,
            (0, 3): 17,
            (1, -2): 18,
            (1, -1): 19,
            (1, 0): 20,
            (1, 1): 21,
            (1, 2): 22,
            (1, 3): 23,
            (2, -2): 24,
            (2, -1): 25,
            (2, 0): 26,
            (2, 1): 27,
            (2, 2): 28,
            (2, 3): 29,
            (3, -2): 30,
            (3, -1): 31,
            (3, 0): 32,
            (3, 1): 33,
            (3, 2): 34,
            (3, 3): 35
        }

        expected_first_to_high = self.revdict(expected_high_to_first)
        found_htf, found_fth   = get_state_mapping(states,2)

        assert_dict_equal(expected_high_to_first,found_htf)
        assert_dict_equal(expected_first_to_high,found_fth)

    def test_third_order(self):
        assert False

    def test_fourth_order(self):
        assert False


def test_transcode_sequences():
    assert False

def test_placeholder():
    assert False
