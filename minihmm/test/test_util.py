#!/usr/bin/env python
import numpy

from numpy.testing import assert_array_equal
from minihmm.test.common import (
    check_list_equal,
    check_array_equal,
    check_tuple_equal,
)
from minihmm.util import (
    matrix_to_dict,
    matrix_from_dict,
    build_transition_table
)


#===============================================================================
# Tests for counters
#===============================================================================


class TestBuildTransitionTable():

    @classmethod
    def setUpClass(cls):
        cls.num_states = 8

        cls.test_seqs = [
            [5, 3, 0, 2, 1, 6, 3, 2, 1, 5, 3, 2, 5, 1, 2, 3, 6],
            [2, 2, 2, 1, 0, 5, 6, 2, 6, 4, 3, 2, 1],
            [6, 2, 4, 2, 4, 5, 0, 2, 4, 6, 4, 3, 7, 2, 2],
        ]

        cls.mats = [
            [[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.],
             [ 0.,  2.,  0.,  1.,  0.,  1.,  0.,  0.],
             [ 1.,  0.,  2.,  0.,  0.,  0.,  1.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  1.,  0.,  2.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

            [[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
             [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  2.,  2.,  0.,  0.,  0.,  1.,  0.],
             [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
             [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

            [[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  3.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
             [ 0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.],
             [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]]

        ]
        cls.mats = [numpy.array(X) for X in cls.mats]

        cls.test_weights = [10, 20, 5]

    def test_no_weights_no_pseudocounts(self):
        expected_counts = sum(self.mats)
        expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
        found_counts = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=0, normalize=False)
        found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=0, normalize=True)

        yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_counts, expected_counts
        yield check_array_equal, found_freqs, expected_freqs

    def test_no_weights_int_pseudocounts(self):
        for pcounts in (1,2,3):
            expected_counts = sum(self.mats) + pcounts
            expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
            found_counts = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=pcounts, normalize=False)
            found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=pcounts, normalize=True)

            yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
            yield check_array_equal, found_counts, expected_counts
            yield check_array_equal, found_freqs, expected_freqs

    def test_no_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, self.num_states))
        expected_counts = sum(self.mats) + pmat
        expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
        found_counts = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=pmat, normalize=False)
        found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=None, pseudocounts=pmat, normalize=True)

        yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_counts, expected_counts
        yield check_array_equal, found_freqs, expected_freqs

    def test_weights_no_pseudocounts(self):
        expected_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            expected_counts += weight*mat

        expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
        found_counts = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=0, normalize=False)
        found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=0, normalize=True)

        yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_counts, expected_counts
        yield check_array_equal, found_freqs, expected_freqs

    def test_weights_int_pseudocounts(self):
        my_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            my_counts += weight*mat

        for pcounts in (1,2,3):
            expected_counts = my_counts + pcounts
            expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
            found_counts = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=pcounts, normalize=False)
            found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=pcounts, normalize=True)

            yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
            yield check_array_equal, found_counts, expected_counts
            yield check_array_equal, found_freqs, expected_freqs

    def test_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, self.num_states))
        expected_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            expected_counts += weight*mat

        expected_counts += pmat
        expected_freqs = (1.0 * expected_counts.T / expected_counts.sum(1)).T
        found_counts = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=pmat, normalize=False)
        found_freqs = build_transition_table(self.num_states, self.test_seqs, weights=self.test_weights, pseudocounts=pmat, normalize=True)

        yield check_tuple_equal, found_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_counts, expected_counts
        yield check_array_equal, found_freqs, expected_freqs


#===============================================================================
# Tests for serialization
#===============================================================================


class TestSerialization():

    @classmethod
    def setUpClass(cls):
        cls.testmat = numpy.array(
            [0.62372267,  0.69672543,  0.5465455 ,  0.        ,  0.32971244,
             0.        ,  0.45617063,  0.        ,  0.10920329,  0.56855817,
             0.91924974,  0.61475372,  0.        ,  0.        ,  0.56096722]).reshape((5,3))

        cls.testdict = {
            "shape" : tuple(cls.testmat.shape),
            "row"   : list(cls.testmat.nonzero()[0]),
            "col"   : list(cls.testmat.nonzero()[1]),
            "data"  : list(cls.testmat[cls.testmat.nonzero()]),

        }

    def test_matrix_to_dict(self):
        found = matrix_to_dict(self.testmat)
        yield check_list_equal, found["row"], list(self.testmat.nonzero()[0])
        yield check_list_equal, found["col"], list(self.testmat.nonzero()[1])
        yield check_array_equal, found["data"], list(self.testmat[self.testmat.nonzero()])

    def test_matrix_from_dict_sparse(self):
        found = matrix_from_dict(self.testdict, dense=False)
        yield check_list_equal, self.testdict["row"], list(found.row)
        yield check_list_equal, self.testdict["col"], list(found.col)
        yield check_array_equal, self.testdict["data"], found.data

    def test_matrix_from_dict_dense(self):
        assert_array_equal(matrix_from_dict(self.testdict, dense=True), self.testmat)


