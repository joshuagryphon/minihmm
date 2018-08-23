#!/usr/bin/env python
import unittest
import numpy
import scipy.sparse

from numpy.testing import assert_array_equal
from minihmm.test.common import (
    check_list_equal,
    check_array_equal,
    check_tuple_equal,
)
from minihmm.util import (
    matrix_to_dict,
    matrix_from_dict,
    build_hmm_tables,
)

#===============================================================================
# Tests for counters
#===============================================================================


class TestBuildHMMTables():
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

        ] # yapf: disable
        cls.mats = [numpy.array(X) for X in cls.mats]

        cls.test_weights = [10, 20, 5]

    def test_transitions_no_weights_no_pseudocounts(self):
        expected_transition_counts = sum(self.mats)
        expected_transition_freqs = (
            1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
        ).T
        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=0,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=0,
            normalize=True
        )

        yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_transition_counts, expected_transition_counts
        yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_no_weights_int_pseudocounts(self):
        for pcounts in (1, 2, 3):
            expected_transition_counts = sum(self.mats) + pcounts
            expected_transition_freqs = (
                1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
            ).T
            found_prior_counts, found_transition_counts = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=None,
                transition_pseudocounts=pcounts,
                normalize=False
            )
            found_prior_freqs, found_transition_freqs = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=None,
                transition_pseudocounts=pcounts,
                normalize=True
            )

            yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
            yield check_array_equal, found_transition_counts, expected_transition_counts
            yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_no_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, self.num_states))
        expected_transition_counts = sum(self.mats) + pmat
        expected_transition_freqs = (
            1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
        ).T
        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=pmat,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=pmat,
            normalize=True
        )

        yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_transition_counts, expected_transition_counts
        yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_weights_no_pseudocounts(self):
        expected_transition_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            expected_transition_counts += weight * mat

        expected_transition_freqs = (
            1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
        ).T
        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            transition_pseudocounts=0,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            transition_pseudocounts=0,
            normalize=True
        )

        yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_transition_counts, expected_transition_counts
        yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_weights_int_pseudocounts(self):
        my_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            my_counts += weight * mat

        for pcounts in (1, 2, 3):
            expected_transition_counts = my_counts + pcounts
            expected_transition_freqs = (
                1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
            ).T
            found_prior_counts, found_transition_counts = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=self.test_weights,
                transition_pseudocounts=pcounts,
                normalize=False
            )
            found_prior_freqs, found_transition_freqs = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=self.test_weights,
                transition_pseudocounts=pcounts,
                normalize=True
            )

            yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
            yield check_array_equal, found_transition_counts, expected_transition_counts
            yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, self.num_states))
        expected_transition_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            expected_transition_counts += weight * mat

        expected_transition_counts += pmat
        expected_transition_freqs = (
            1.0 * expected_transition_counts.T / expected_transition_counts.sum(1)
        ).T
        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            transition_pseudocounts=pmat,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            transition_pseudocounts=pmat,
            normalize=True
        )

        yield check_tuple_equal, found_transition_counts.shape, (self.num_states, self.num_states)
        yield check_array_equal, found_transition_counts, expected_transition_counts
        yield check_array_equal, found_transition_freqs, expected_transition_freqs

    def test_transitions_alternate_initializer(self):
        expected_transition_counts = 0
        for mat, weight in zip(self.mats, self.test_weights):
            expected_transition_counts += weight * mat
            found_prior_counts, found_transition_counts = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=self.test_weights,
                normalize=False,
                initializer=scipy.sparse.dok_matrix
            )

        found_dense = found_transition_counts.todense()
        assert_array_equal(found_dense, expected_transition_counts)

    def test_state_priors_no_weights_no_pseudocounts(self):
        expected_prior_counts = numpy.zeros(self.num_states)
        for my_seq in self.test_seqs:
            expected_prior_counts[my_seq[0]] += 1

        expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=0,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            transition_pseudocounts=0,
            normalize=True
        )

        yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
        yield check_array_equal, found_prior_counts, expected_prior_counts
        yield check_array_equal, found_prior_freqs, expected_prior_freqs

    def test_state_priors_no_weights_int_pseudocounts(self):
        my_counts = numpy.zeros(self.num_states)
        for my_seq in self.test_seqs:
            my_counts[my_seq[0]] += 1

        for pcounts in (1, 2, 3):
            expected_prior_counts = my_counts + pcounts
            expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

            found_prior_counts, found_transition_counts = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=None,
                state_prior_pseudocounts=pcounts,
                normalize=False
            )
            found_prior_freqs, found_transition_freqs = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=None,
                state_prior_pseudocounts=pcounts,
                normalize=True
            )

            yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
            yield check_array_equal, found_prior_counts, expected_prior_counts
            yield check_array_equal, found_prior_freqs, expected_prior_freqs

    def test_state_priors_no_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, ))
        my_counts = 0
        my_counts = numpy.zeros(self.num_states)
        for my_seq in self.test_seqs:
            my_counts[my_seq[0]] += 1

        expected_prior_counts = my_counts + pmat
        expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            state_prior_pseudocounts=pmat,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=None,
            state_prior_pseudocounts=pmat,
            normalize=True
        )

        yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
        yield check_array_equal, found_prior_counts, expected_prior_counts
        yield check_array_equal, found_prior_freqs, expected_prior_freqs

    def test_state_priors_weights_no_pseudocounts(self):
        expected_prior_counts = numpy.zeros(self.num_states)
        for (my_seq, my_weight) in zip(self.test_seqs, self.test_weights):
            expected_prior_counts[my_seq[0]] += my_weight

        expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states, self.test_seqs, weights=self.test_weights, normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states, self.test_seqs, weights=self.test_weights, normalize=True
        )

        yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
        yield check_array_equal, found_prior_counts, expected_prior_counts
        yield check_array_equal, found_prior_freqs, expected_prior_freqs

    def test_state_priors_weights_int_pseudocounts(self):
        my_counts = numpy.zeros(self.num_states)
        for (my_seq, my_weight) in zip(self.test_seqs, self.test_weights):
            my_counts[my_seq[0]] += my_weight

        for pcounts in (1, 2, 3):
            expected_prior_counts = my_counts + pcounts
            expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

            found_prior_counts, found_transition_counts = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=self.test_weights,
                state_prior_pseudocounts=pcounts,
                normalize=False
            )
            found_prior_freqs, found_transition_freqs = build_hmm_tables(
                self.num_states,
                self.test_seqs,
                weights=self.test_weights,
                state_prior_pseudocounts=pcounts,
                normalize=True
            )

            yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
            yield check_array_equal, found_prior_counts, expected_prior_counts
            yield check_array_equal, found_prior_freqs, expected_prior_freqs

    def test_state_priors_weights_array_pseudocounts(self):
        pmat = numpy.random.randint(0, high=255, size=(self.num_states, ))
        my_counts = numpy.zeros(self.num_states)
        for my_seq, my_weight in zip(self.test_seqs, self.test_weights):
            my_counts[my_seq[0]] += my_weight

        expected_prior_counts = my_counts + pmat
        expected_prior_freqs = (1.0 * expected_prior_counts) / expected_prior_counts.sum()

        found_prior_counts, found_transition_counts = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            state_prior_pseudocounts=pmat,
            normalize=False
        )
        found_prior_freqs, found_transition_freqs = build_hmm_tables(
            self.num_states,
            self.test_seqs,
            weights=self.test_weights,
            state_prior_pseudocounts=pmat,
            normalize=True
        )

        yield check_tuple_equal, found_prior_counts.shape, (self.num_states, )
        yield check_array_equal, found_prior_counts, expected_prior_counts
        yield check_array_equal, found_prior_freqs, expected_prior_freqs


#===============================================================================
# Tests for serialization
#===============================================================================


class TestSerialization():
    @classmethod
    def setUpClass(cls):
        cls.testmat = numpy.array(
            [0.62372267,  0.69672543,  0.5465455 ,  0.        ,  0.32971244,
             0.        ,  0.45617063,  0.        ,  0.10920329,  0.56855817,
             0.91924974,  0.61475372,  0.        ,  0.        ,  0.56096722]).reshape((5,3)) # yapf: disable

        cls.testdict = {
            "shape": tuple(cls.testmat.shape),
            "row": list(cls.testmat.nonzero()[0]),
            "col": list(cls.testmat.nonzero()[1]),
            "data": list(cls.testmat[cls.testmat.nonzero()]),
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
