#!/usr/bin/env python
"""Utilities for changing representations of observation sequences and/or models.
The primary functionality of the tools in this module is to translate Nth-order
HMMs into equivalent first order HMMs.


Notes on model reduction
------------------------
In a high-order model, the ends of sequences introduce inhomogeneities, similar
to state priors in first-order models. For example, in a 3rd order model, the
first two states must be handled separately from those remaining:

.. math::

   P = p(S_1) * p(S_2 | S_1) * p(S_3 | S_2, S_1) * p(S_4 | S_3, S_2) *  ... * p(S_N |, S_{N-2}, S_{N-1})

Rather than adding extra probability tables to represent these inhomogeneities,
:mod:`minihmm` creates dummy states. For a third-order model, we would need
to add two dummy start states in high-order space, $ S_{dummy1} $ and $ S_{dummy2} $.

We rephrase the model above as:

.. math::

   P = p(S_1 | S_{dummy2}, S_{dummy1}) * p(S_2 | S_1, S_{dummy2}, ) * p(S_3 | S_2, S_1) * p(S_4 | S_3, S_2) *  ... * p(S_N |, S_{N-2}, S_{N-1})

Then:

 - The prior probabilities of every state pair are set to zero, except
   $ p(S_{dummy2}, S_{dummy1}) $, which is  set to one.

 - The transition probabilities of p(S_i | S_{dummy2}, S_{dummy1}) are set to
   the prior probabilities of P(S_i) in the previous model description

 - Transition probabilities of p(S_i | S_{i-1}, S_{dummy2}) are set to the
   observed values for P(S_i | S_{i-1}) in the previous model description

:mod:`minihmm` contains tools for creating these dummy states, and remapping
state sequences accordingly.



Workflow
--------

A number of different workflows are possible depending upon what type of
starting information is available:


If state path for each observation sequence are known
.....................................................

In this case, it is easiest to transcode the state sequences into first-order
space, and then calculate the model parameters from the known state sequences.
To do this, call :meth:`ModelReducer.lower_stateseq_orders`, to re-map state
sequences into the reduced model space, then calculate transition probabilities
as for a first order model.


If series of states are unknown, and training is required
.........................................................

In this case, instantiate a :class:`~minihmm.represent.ModelReducer` and either
(1a) create a naive model using
:meth:`~minihmm.represent.ModelReducer.get_random_model`, or, better, (1b) remap
parameters from a related first-order model using
:meth:`~minihmm.represent.ModelReducer.remap_from_first_order` and then (2)
train the resulting HMM using standard method (e.g.
:func:`~minihmm.training.train_baum_welch`), tying emission factors to improve
fitting (see :meth:`~minihmm.represent.ModelReducer.get_emission_mapping`.



"""
import warnings
import itertools
import copy
import numpy

import jsonpickle
import jsonpickle.ext.numpy
jsonpickle.ext.numpy.register_handlers()

from minihmm.hmm import FirstOrderHMM
from minihmm.factors import ArrayFactor, MatrixFactor

from scipy.sparse import (lil_matrix, dok_matrix, coo_matrix)

#===============================================================================
# Model translation
#===============================================================================


def _get_modelreducer_from_dict(dtmp):
    """Revive a :class:`ModelReducer` from a dictionary made by
    :meth:`~ModelReducer._to_dict`

    Parameters
    ----------
    dtmp : dict
        Dictionary exported by :meth:`~ModelReducer._to_dict`
        
    Returns
    -------
    ModelReducer
        Revived model
    """
    # code note: logic has to be in an independent function as opposed
    # to a static method in order to enable its use in __reduce__
    return ModelReducer(dtmp["starting_order"], dtmp["num_states"], hmm=dtmp.get("hmm", None))


class ModelReducer(object):
    """Utility class for reducing high-order HMMs to equivalent first-order HMMs.


    Attributes
    ----------
    starting_order : int
        Order of starting model

    high_order_states : int
        Number of states, in high-order space

    low_order_states : int
        Number of states, in fist-order space

    high_states_to_low : dict
        Dictionary mapping high-order states to tuples of low-order states

    low_states_to_high : dict
        Dicitonary mapping tuples of low-order states to high-order states

    hmm : :class:`~minihmm.hmm.FirstOrderHMM`
        Associated first-order HMM. At the moment, this must be constructed by
        the user.. Will be used for sampling, decoding, et c
    """

    def __init__(self, starting_order, num_states, hmm=None):

        if num_states < 1:
            raise ValueError("Must have >= 1 state in model.")
        if starting_order < 1:
            raise ValueError("Cannot reduce order of 0-order model")
        elif starting_order == 1:
            warnings.warn("Reducing a first-order model doesn't make much sense.", UserWarning)

        self.starting_order = starting_order
        self.high_order_states = num_states
        self._dummy_states = self._get_dummy_states()
        self.high_states_to_low, self.low_states_to_high = self._get_state_mapping()
        self.low_order_states = len(self.low_states_to_high)
        self._hmm = hmm

    @property
    def hmm(self):
        if self._hmm is None:
            cname = self.__class__.__name__
            raise ValueError(
                "No HMM associated with %s. Please set `this.hmm` to something." % cname
            )

        return self._hmm

    @hmm.setter
    def hmm(self, value):
        if not isinstance(value, FirstOrderHMM) and value is not None:
            raise ValueError(
                "`hmm` must be a valid FirstOrderHMM. Instead got type '%s'" %
                (type(value).__name__)
            )

        self._hmm = value

    def __str__(self):
        return "<%s order=%s high_states=%s hmm=%s>" % (
            self.__class__.__name__, self.starting_order, self.high_order_states,
            self._hmm is not None
        )

    def __eq__(self, other):
        """Test whether `self` is equal to `other`, defined as equality of:

         - number of high order states
         - starting model order
         - if an HMM is defined for either `self` or `other`, both must be defined,
           and have equal parameters

        Returns
        -------
        bool
            `True` if `self` equals `other`, otherwise `False`
        """
        if self.starting_order != other.starting_order \
           or self.high_order_states != self.high_order_states \
           or self.low_order_states != other.low_order_states:
            return False

        if self._hmm is None:
            if other._hmm is None:
                return True
        else:
            if other._hmm is None:
                return False
            return self.hmm == other.hmm

    def __repr__(self):
        return str(self)

    def __reduce__(self):
        """Define pickling and unpickling methods for `self`"""
        return _get_modelreducer_from_dict, (self._to_dict(), )

    def _to_dict(self):
        """Convenience method to export minimal elements required for pickling

        Returns
        -------
        dict
            Dictionary representation of `self`
        """
        dtmp = {
                "starting_order" : self.starting_order,
                "num_states"     : self.high_order_states,
        } # yapf: disable
        if self._hmm is not None:
            dtmp["hmm"] = self._hmm

        return dtmp

    @staticmethod
    def _from_dict(dtmp):
        """Revive a :class:`ModelReducer` from a dictionary made by
        :meth:`ModelReducer._to_dict`

        Returns
        -------
        ModelReducer
            Revived model
        """
        # code note: logic has to be in an independent function as opposed
        # to a static method in order to enable its use in __reduce__
        return _get_modelreducer_from_dict(dtmp)

    def to_json(self):
        """Convert `self` to a JSON blob

        Returns
        -------
        str
            JSON blob of `self`
        """
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(stmp):
        """Revive a :class:`ModelReducer` from a string-format JSON blob"""
        return jsonpickle.decode(stmp)

    @staticmethod
    def transcode_sequence(sequence, alphadict):
        """Transcode a single sequence from one alphabet to another

        Parameters
        ----------
        sequence : list-like
            A sequence

        alphadict : dict-like
            Dictionary mapping symbols in input sequence to symbols in output
            sequence

        Returns
        -------
        :class:`numpy.ndarray`
            Transcoded sequence
        """
        return numpy.array([alphadict[X] for X in sequence])

    @staticmethod
    def transcode_sequences(sequences, alphadict):
        """Transcode a set of sequences from one alphabet to another

        Parameters
        ----------
        sequences : iterable of list-like
            Iterable of sequences (each, itself, an iterable like a list et c)

        alphadict : dict-like
            Dictionary mapping symbols in input sequence to symbols in output
            sequence

        Returns
        -------
        list of :class:`numpy.ndarray`
            List of transcoded sequences
        """
        return [ModelReducer.transcode_sequence(X, alphadict) for X in sequences]

    def _get_dummy_states(self):
        """Create sorted lists of dummy states required to reduce `self` to
        an equivalent first-order model.

        By convention, dummy states are given negative indices in high-order
        space.  Don't rely on this- it may change in the future

        Parameters
        ----------
        starting_order : int
            Starting order of HMM/MM

        Returns
        -------
        list
            List of new dummy states, sorted
        """
        return list(range(1 - self.starting_order, 0))

    def _get_state_mapping(self):
        """Create dicts mapping states between a high-order and a first-order
        HMM.

        Returns
        -------
        :class:`dict`
            Forward state map, mapping high-order states to tuples of
            equivalent low-order states

        :class:`dict`
            Reverse state map, mapping new first-order states to tuples of
            high-order states
        """
        forward = {}
        reverse = {}
        states = list(range(self.high_order_states))

        c = 0

        # create maps for newly added start and end states
        for idx in range(self.starting_order - 1):
            root = self._dummy_states[idx:]
            for symbol in itertools.product(*([states] * (idx + 1))):
                tup = tuple(root + list(symbol))
                forward[tup] = c
                reverse[c] = tup
                c += 1

        # combinations of true states
        for n, symbol in enumerate(itertools.product(*([states] * self.starting_order))):
            forward[symbol] = n + c
            reverse[c + n] = symbol

        return forward, reverse

    def _get_stateseq_tuples(self, state_seqs):
        """Remap a high-order sequence of states into tuples for use in a
        low-order model, adding dummy start states

        Notes
        -----
        This does *not* remap those tuples into lower state spaces. Use
        :func:`lower_stateseq_orders`, which wraps this function, for that


        Parameters
        ----------
        state_seqs : list of list-like
            List of state sequences

        Returns
        -------
        list of lists
            List of tuples of state sequences
        """
        dummy_states = self._dummy_states
        starting_order = self.starting_order
        outseqs = []
        for n, inseq in enumerate(state_seqs):
            if (numpy.array(inseq) < 0).any():
                raise ValueError("Found negative state label in input sequence %s!" % n)

            baseseq = dummy_states + list(inseq)
            outseqs.append(
                [
                    tuple(baseseq[idx:idx + starting_order])
                    for idx in range(0,
                                     len(baseseq) - starting_order + 1)
                ]
            )

        return outseqs

    def lower_stateseq_orders(self, state_seqs):
        """Map a high-order sequence of states into an equivalent first-order
        state sequence, creating dummy states as necessary and dictionaries
        that map states between high and first-order spaces.


        Parameters
        ----------
        state_seqs : list of list-like
            List of state sequences, each given in high-order space

        Returns
        -------
        list of :class`numpy.ndarray`
            List of state sequences, represented in first-order space


        See also
        --------
        raise_stateseq_orders
        """
        dummy_states = self._dummy_states
        state_map = self.high_states_to_low

        tuple_seqs = self._get_stateseq_tuples(state_seqs)
        return ModelReducer.transcode_sequences(tuple_seqs, state_map)

    def raise_stateseq_orders(self, state_seqs):
        """Map a state sequence from first-order space back to original
        high-order space

        Parameters
        ----------
        state_seqs : list
            List of high-order state sequences

        Returns
        -------
        list of :class:`numpy.ndarray`
            State sequences, in high-order space


        See also
        --------
        lower_stateseq_orders
        """
        ltmp = []
        for t in self.transcode_sequences(state_seqs, self.low_states_to_high):
            ltmp.append([X[-1] for X in t])

        return ltmp

    def viterbi(self, emissions):
        """Finds the most likely state sequence underlying a set of emissions
        using the Viterbi algorithm.

        See http://en.wikipedia.org/wiki/Viterbi_algorithm

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        Returns
        -------
        `viterbi_states`
            :class:`numpy.ndarray`. Decoded labels for each position in
            `emissions[start:end]`
        """
        raw = self.hmm.viterbi(emissions)["viterbi_states"]
        high = self.raise_stateseq_orders([raw])[0]
        return high

    def posterior_decode(self, emissions):
        """Find the most probable state for each individual state in the
        sequence of emissions, using posterior decoding. Note, this objective
        is distinct from finding the most probable sequence of states for all
        emissions, as is given in Viterbi decoding. This alternative may be
        more appropriate when multiple paths have similar probabilities.


        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations


        Returns
        -------
        numpy.ndarray
            An array of dimension `[t x 1]` of the most likely states at each point `t`
        """
        raw, _ = self.hmm.posterior_decode(emissions)
        return self.raise_stateseq_orders([raw])[0]

    def sample(self, emissions, num_samples):
        """Sample state sequences from the distribution P(states | emissions),
        by tracing backward through the matrix of forward probabilities.
        See Durbin1997 ch 4.3, section "Probabilistic sampling of aligments"

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        num_samples : int, optional
            Number of state paths to generate (Default: 1)


        Returns
        -------
        list
            List of state paths
        """
        raw_paths = self.hmm.sample(emissions, num_samples=num_samples)
        return self.raise_stateseq_orders(raw_paths)

    def generate(self, length):
        """Generates a random sequence of states and emissions from the HMM

        Parameters
        ----------
        length : int
            Length of sequence to generate


        Returns
        -------
        numpy.ndarray
            Array of dimension `[t x 1]` indicating the HMM state at each
            timestep

        numpy.ndarray
            Array of dimension `[t x Q]` indicating the observation at each
            timestep.  `Q = 1` for univariate HMMs, or more than 1 if
            observations are multivariate.

        float
            Joint log probability of generated state and observation sequence.
            **Note**: this is different from the log probability of the
            observation sequence alone, which would be the sum of its joint
            probabilities with all possible state sequences.


        Notes
        -----
        The HMM can only generate sequences if all of its EmissionFactors are
        generative. I.e. if using :class:`minihmm.factors.FunctionFactor` or
        :class:`minihmm.factors.LogFunctionFactor` , generator functions must
        be specified at their instantiation.
        """
        raw_path, obs, logprob = self.hmm.generate(length)
        high_path = self.raise_stateseq_orders([raw_path])[0]
        return high_path, obs, logprob

    def joint_path_logprob(self, path, emissions):
        """Return log P(path, emissions) evaluated under this model

        Parameters
        ----------
        path : list-like
            Sequence of states

        emissions : list-like
            Sequence of observations

        Returns
        -------
        float
            Log probability of P(path, emissions)
        """
        lower_path = self.lower_stateseq_orders([path])[0]
        return self.hmm.joint_path_logprob(lower_path, emissions)

    def remap_emission_factors(self, emission_probs):
        """Map emission probabilities from high-order space to equivalent
        reduced space

        Parameters
        ----------
        emission_probs : list of :class:`~minihmm.factors.AbstractFactor`
            List of emission probabilities, indexed by state in high-order
            space.

        Returns
        -------
        list
            list of Factors, indexed by states in equivalent first-order
        """
        # make empty list length of new states
        ltmp = [None] * len(self.low_states_to_high)

        # populate list
        for newstate, state_tuple in self.low_states_to_high.items():
            ltmp[newstate] = emission_probs[state_tuple[-1]]

        return ltmp

    # convert to csr, csc, or dense before computations depending on which needed
    # serialize coomat as coomat.row, coomat.col, coomat.data
    def get_pseudocount_arrays(self):
        """Return a valid pseudocount array for state priors and for transition
        tables in first-order space, where *valid* stipulates that cells
        corresponding transitions that cannot exist in high-order space are set
        to zero.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of state prior pseudocounts

        :class:`scipy.sparse.coo_matrix`
            Sparse matrix of transition pseudocounts
        """
        state_priors = numpy.zeros(self.low_order_states)
        for i in range(self.high_order_states):
            state_priors[self.high_states_to_low[tuple(self._dummy_states + [i])]] = 1

        row_ords = []
        col_ords = []
        for lstate, hseq in self.low_states_to_high.items():
            stub = hseq[1:]
            for i in range(self.high_order_states):
                next_state = self.high_states_to_low[tuple(list(stub) + [i])]
                row_ords.append(lstate)
                col_ords.append(next_state)

        vals = [1] * len(row_ords)

        return state_priors, coo_matrix((vals, (row_ords, col_ords)))

    def get_emission_mapping(self):
        """Generate an array mapping emission factors in translated low-order
        models to their corresponding emission factors in native, high-order
        states. In training, the parameters for these groups of emissions
        should be tied, as they are equivalent.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of length `len(self.high_states_to_low)`, indexed by state in
            low-order space, mapping low-order states to corresponding native
            high-order states

        """
        # can rely on lexicographic pairing of high-order state tuples
        # and translated, low-order state numbers to produce correct tying vector
        return numpy.array([X[-1] for X in sorted(self.high_states_to_low)])


#    # TODO
#    def get_random_model(self):
#        """
#        See also
#        --------
#        :meth:`ModelReducer.remap_from_first_order`
#        """
#        pass

    def remap_from_first_order(self, native_hmm):
        """Remap parameters from a native first order HMM onto a first-order
        translation of a high-order HMM, in order to, for example, provide a
        reasonable non-random starting point for refinement training of the
        high-order HMM.

        Parameters
        ----------
        native_hmm : :class:`minihmm.hmm.FirstOrderHMM`
            Native, first-order HMM, preferably with trained parameters

        Returns
        -------
        :class:`~minihmm.hmm.FirstOrderHMM`
            First-order representation of the high-order HMM structure
            described by `self`, with parameters from `native_hmm` remapped
            into corresponding positions.
        """
        htl = self.high_states_to_low

        # check that number of states is compatible
        if self.high_order_states != native_hmm.num_states:
            raise ValueError(
                "Native HMM (%d states), has different number of states than `self` (%d states)" %
                (native_hmm.num_states, self.high_order_states)
            )

        # For transitions
        # Each high-order state transitiono  `(n-i, ... , n-1) ->  (n-i+1 , ... , n)`
        # should be mapped to appropiate transformations of the parameters (n - 1 , n)

        # For state priors and emission probabilities
        # each high-order state (n-i, ...,  n) should be given the parameters matching
        # native state `n`

        # will need to make appropriate state-tying matrices for emissions, as well
        sp_source = native_hmm.state_priors.data
        sp_dest = numpy.zeros(self.low_order_states, dtype=float)

        trans_source = native_hmm.trans_probs.data
        trans_dest = numpy.zeros((self.low_order_states, self.low_order_states), dtype=float)

        em_source = native_hmm.emission_probs
        em_dest = [None] * self.low_order_states

        for my_tuple, trans_state in htl.items():
            native_state = my_tuple[-1]
            sp_dest[trans_state] = sp_source[native_state]
            em_dest[trans_state] = copy.deepcopy(em_source[native_state])

            for next_native_state in range(self.high_order_states):
                next_tuple = tuple(list(my_tuple)[1:] + [next_native_state])
                next_trans_state = htl[next_tuple]
                trans_dest[trans_state, next_trans_state] = trans_source[native_state,
                                                                         next_native_state]

        # renormalize
        sp_dest /= sp_dest.sum()
        sp_dest = ArrayFactor(sp_dest)

        # shoudln't have to renormalize; check this
        trans_dest = (trans_dest.T / trans_dest.sum(1)).T
        trans_dest = MatrixFactor(trans_dest)

        return FirstOrderHMM(state_priors=sp_dest, emission_probs=em_dest, trans_probs=trans_dest)
