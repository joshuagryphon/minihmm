#!/usr/bin/env python
"""Estimator classes for Baum-Welch training. Estimators determine how
parameters for the HMM are re-estimated from observations during each training
cycle.

Estimators must be able to:

    1. Determine whether a set of observations is valid under the probability
       distribution. This is implemented by the methods ``is_valid()`` and
      ``is_invalid()``

    2. Reduce a set of observations to expected summary statistics (expectation
       step), implemented by the method ``reduce_data()``

    3. Estimate improved parameters for the model from the expected statistics
    (maximization step), and, using these parameters, initialize new Factors.
    These steps are implemented by the method ``construct_factors()``

Here we provide examples of estimators for discrete and continuous
distributions, as well as "frozen" estimators which keep parameter values
constant through Baum-Welch training.
"""

from minihmm.factors import (
    ArrayFactor,
    MatrixFactor,
    FunctionFactor,
    LogFunctionFactor,
    ScipyDistributionFactor,
)

from abc import abstractmethod
import numpy
import scipy.stats
import copy

#===============================================================================
# INDEX: helper functions
#===============================================================================


def get_model_noise(template, weight, assymetric_weights=1):
    """Create an array of noise,  with total number of counts equal to
    `weight*template.sum()` and shape equal to template.shape

    Parameters
    ----------
    template : numpy.ndarray
        Array indicating where to put noise, and in what relative distributions

    weight : float
        Relative weight of noise relative to data

    pseudocount_weights : float or numpy.ndarray
        If given, assymetric weights of pseudocounts to noise (e.g. set cells
        to zero to prevent noise from being added to those regions)

    Returns
    -------
    :class:`numpy.ndarray`
        Numpy array containing noise
    """
    noise = numpy.random.random(template.shape) * assymetric_weights
    noise = noise / noise.sum() * weight * template.sum()
    return noise


#===============================================================================
# INDEX: estimators
#===============================================================================


class AbstractProbabilityEstimator(object):
    """Helper class for reestimation of probabilities in Baum-Welch training.
    Subclasses will be used during training to extract sufficient statistics
    from observation data (via calls to
    :meth:`AbstractProbabilityEstimator.reduce_data`, and to create new factors
    from those statistics (via calls to
    :meth:`AbstractProbabilityEstimator.construct_factors`)
    """

    @abstractmethod
    def is_valid(self, reduced_data):
        """Return true if data is valid and should be included in current round
        of reestimation

        Parameters
        ----------
        reduced_data
            output from :meth:`AbstractProbabilityEstimator.reduce_data`
        """
        pass

    def is_invalid(self, reduced_data):
        """Return true if data is invalid and should be excluded from current
        round of reestimation

        Parameters
        ----------
        reduced_data
            output from :meth:`AbstractProbabilityEstimator.reduce_data``
        """
        return not self.is_valid(reduced_data)

    @abstractmethod
    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Collect data from a single observation sequence and reduce it to a form
        amenable for factor construction by
        :meth:`AbstractProbabilityEstimator.construct_factors`

        Parameters
        ----------
        my_obs : list-like
            Observation sequence

        obs_logprob : float
            Observation logprob, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        forward : numpy.ndarray
            Scaled forward probability matrix, from
            :meth:`minihmm.hmm.FirstOrderHMM.forward_backward`

        backward : numpy.ndarray
            Scaled backward probability matrix, from
            :meth:`minihmm.hmm.FirstOrderHMM.forward_backward`

        scale_factors : numpy.ndarray
            Scale factors used in scaling `forward` and `backward` to
            floating-point friendly sizes, from
            :meth:`minihmm.hmm.FirstOrderHMM.forward_backward`

        ksi : numpy.ndarray
            `MxNxT` array describing the full probability of being in state `M`
            at time `t` and state `N` at time `t+1`. From
            :meth:`minihmm.hmm.FirstOrderHMM.forward_backward`

        Returns
        -------
        numpy.ndarray
            Array of appropirate shape containing sufficient statistics
        """
        pass

    @abstractmethod
    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct factor(s) for an HMM using reduced data from observation
        sequences

        Parameters
        ----------
        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`AbstractProbabilityEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations
            (e.g. transition counts, state prior counts, emission counts, et c)
            in data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of observations
            (transition counts, state prior counts, emission counts, et c)
            in data set (Default: 1e-8)

        Returns
        -------
        some sort of probability factor
        """
        pass


class _FrozenParameterEstimator(AbstractProbabilityEstimator):
    """Keep some parameter constant, completely ignoring observations
    """

    def is_valid(self, reduced_data):
        return True

    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Completely ignore data,  since parameter is frozen"""
        return None


class FrozenStatePriorEstimator(_FrozenParameterEstimator):
    """Keep state priors constant, completely ignoring observation data
    """

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Return model's previous state prior factor"""
        return model.state_priors


class FrozenTransitionEstimator(_FrozenParameterEstimator):
    """Keep transition estimates constant,  completely ignoring observation data
    """

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Return model's previous transition factor"""
        return model.trans_probs


class FrozenEmissionEstimator(_FrozenParameterEstimator):
    """Keep emission estimates constant,  completely ignoring observation data
    """

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Return model's previous emission factors"""
        return model.emission_probs


class _DiscreteParameterEstimator(AbstractProbabilityEstimator):
    """Estimator that models CPDs as tables
    """

    def is_valid(self, reduced_data):
        return not (numpy.isnan(reduced_data) | numpy.isinf(reduced_data)).any()


class DiscreteStatePriorEstimator(_DiscreteParameterEstimator):
    """Estimate discrete state priors from observation sequences in Baum-Welch
    training, modeling these as an :class:`~minihmm.factors.ArrayFactor`
    """

    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Collect data from a single observation sequence and reduce it to a
        form amenable for factor construction

        Parameters
        ----------
        my_obs : list-like
            Observation sequence

        obs_logprob : float
            Observation logprob, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        forward : numpy.ndarray
            Scaled forward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        backward : numpy.ndarray
            Scaled backward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        scale_factors : numpy.ndarray
            Scale factors used in scaling `forward` and `backward` to
            floating-point friendly sizes, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        ksi : numpy.ndarray
            `MxNxT` array describing the full probability of being in state `M`
            at time `t` and state `N` at time `t+1`. From
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`


        Returns
        -------
        numpy.ndarray
            State prior matrix for given observation sequence
        """
        return ksi[0, :, :].sum(1)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete transition factor for an HMM using reduced data
        from observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`DiscreteStatePriorEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        :class:`~minihmm.factors.ArrayFactor`
            State prior probability factor
        """
        pi = sum(reduced_data)
        pi_sum = pi.sum()
        pi += get_model_noise(pi, noise_weight)
        pi += (pseudocount_weight * pi_sum / len(pi))
        pi_normed = pi / pi.sum()
        state_priors = ArrayFactor(pi_normed)
        return state_priors


class DiscreteTransitionEstimator(_DiscreteParameterEstimator):
    """Estimate transitions between states from observation sequences in Baum-Welch
    training. Constructs a MatrixFactor
    """

    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Collect data from a single observation sequence and reduce it to a form
        amenable for factor construction

        Parameters
        ----------
        my_obs : list-like
            Observation sequence

        obs_logprob : float
            Observation logprob, from
            :meth:`minihmm.hmm.FirstOrderHMM.forward_backward`

        forward : numpy.ndarray
            Scaled forward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        backward : numpy.ndarray
            Scaled backward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        scale_factors : numpy.ndarray
            Scale factors used in scaling `forward` and `backward` to
            floating-point friendly sizes, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        ksi : numpy.ndarray
            `MxNxT` array describing the full probability of being in state `M`
            at time `t` and state `N` at time `t+1`. From
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`


        Returns
        -------
        numpy.ndarray
            Transition matrix for given observation sequence
        """
        return ksi.sum(0)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete transition factor for an HMM using reduced data
        from observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`DiscreteTransitionEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        :class:`~minihmm.factors.MatrixFactor`
            Transition probability factor
        """
        A = sum(reduced_data)
        A_sum = A.sum()
        A += get_model_noise(A, noise_weight)
        A += (pseudocount_weight * A_sum / len(A.ravel()))
        A_normed = (A.T / A.sum(1)).T
        return MatrixFactor(A_normed)


class DiscreteEmissionEstimator(_DiscreteParameterEstimator):
    """Estimate discrete emissions from observation sequences in Baum-Welch
    training, modeling these as a series of ArrayFactos
    """

    def __init__(self, num_symbols):
        """Create a DiscreteEmissionEstimator

        Parameters
        ----------
        num_symbols : int
            Number of symbols that can be emitted
        """
        self.num_symbols = num_symbols
        _DiscreteParameterEstimator.__init__(self)

    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Collect data from a single observation sequence and reduce it to a
        form amenable for factor construction

        Parameters
        ----------
        my_obs : list-like
            Observation sequence

        obs_logprob : float
            Observation logprob, from :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        forward : numpy.ndarray
            Scaled forward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        backward : numpy.ndarray
            Scaled backward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        scale_factors : numpy.ndarray
            Scale factors used in scaling `forward` and `backward` to
            floating-point friendly sizes, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        ksi : numpy.ndarray
            `MxNxT` array describing the full probability of being in state `M`
            at time `t` and state `N` at time `t+1`. From
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        Returns
        -------
        numpy.ndarray
            Emission matrix for given observation sequence
        """
        num_states = ksi.shape[1]  # ksi is time x states x states

        my_E = numpy.zeros((num_states, self.num_symbols), dtype=float)
        postprob = forward * backward
        tmp = postprob * scale_factors[:, None]
        for i in range(num_states):
            for k in range(self.num_symbols):
                my_E[i, k] += tmp[my_obs == k, i].sum()

        return my_E

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete emission factor for an HMM using reduced data from
        observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`DiscreteEmissionEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        list
            list of :class:`~minihmm.factors.ArrayFactor` objects, representing
            emission probabilities for each state
        """
        E = sum(reduced_data)
        E_sum = E.sum()
        E += get_model_noise(E, noise_weight)
        E += (pseudocount_weight * E_sum / len(E.ravel()))
        E_normed = (E.T / E.sum(1)).T
        emission_factors = []
        for i in range(E_normed.shape[0]):
            emission_factors.append(ArrayFactor(E_normed[i, :]))

        return emission_factors


class PseudocountStatePriorEstimator(DiscreteStatePriorEstimator):
    """Abstract class. Subclass and define `cls.pseudocount_array` to estimate
    state priors using arbitrarily distributed pseudocounts in Baum-Welch
    training
    """

    def __init__(self, pseudocount_array):
        """
        model : :class:`~minihmm.hmm.FirstOrderHMM`
            HMM to which this estimator will be attached

        pseudocount_array : float or numpy.ndarray
            Scalar (if evenly applying pseudocounts) or numpy array (if
            assymetrically weighting pseudocounts) pseudocounts to apply during
            estimation.
        """
        self.pseudocount_array = copy.deepcopy(pseudocount_array)
        DiscreteStatePriorEstimator.__init__(self)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete state prior factor for an HMM using reduced data
        from observation sequences

        Parameters
        ----------
        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`PseudocountStatePriorEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of observations
            (transition counts, state prior counts, emission counts, et c)
            in data set (Default: 1e-8)

        Returns
        -------
        :class:`~minihmm.factors.MatrixFactor`
            Transition probability factor
        """
        pi = sum(reduced_data)
        pi_sum = pi.sum()
        pi += get_model_noise(pi, noise_weight, assymetric_weights=self.pseudocount_array)
        pi += (pseudocount_weight * pi_sum * self.pseudocount_array) / self.pseudocount_array.sum()
        pi_normed = pi / pi.sum()
        state_priors = ArrayFactor(pi_normed)
        return state_priors


class PseudocountTransitionEstimator(DiscreteTransitionEstimator):
    """Abstract class. Subclass and define `cls.pseudocount_array` to estimate
    state transitions using arbitrarily distributed pseudocounts in Baum-Welch
    training
    """

    def __init__(self, pseudocount_array):
        """
        Parameters
        ----------
        pseudocount_array : float or numpy.ndarray
            Scalar (if evenly applying pseudocounts) or numpy array (if
            assymetrically weighting pseudocounts) pseudocounts to apply during
            estimation.
        """
        self.pseudocount_array = copy.deepcopy(pseudocount_array)
        DiscreteTransitionEstimator.__init__(self)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete transition factor for an HMM using reduced data
        from observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`PseudocountTransitionEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        :class:`~minihmm.factors.MatrixFactor`
            Transition probability factor
        """
        A = sum(reduced_data)
        A_sum = A.sum()
        A += get_model_noise(A, noise_weight, assymetric_weights=self.pseudocount_array)
        A += (pseudocount_weight * A_sum * self.pseudocount_array) / self.pseudocount_array.sum()
        A_normed = (A.T / A.sum(1)).T
        return MatrixFactor(A_normed)


class PseudocountEmissionEstimator(DiscreteEmissionEstimator):
    """Abstract class. Subclass and define `cls.pseudocount_array` to estimate
    emissions using arbitrarily distributed pseudocounts in Baum-Welch training
    """

    def __init__(self, model, num_symbols, pseudocount_array):
        """
        model : :class:`~minihmm.hmm.FirstOrderHMM`
            HMM to which this estimator will be attached

        pseudocount_array : float or numpy.ndarray
            Scalar (if evenly applying pseudocounts) or numpy array (if
            assymetrically weighting pseudocounts) pseudocounts to apply during
            estimation.
        """
        self.pseudocount_array = copy.deepcopy(pseudocount_array)
        DiscreteEmissionEstimator.__init__(self, model, num_symbols)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct discrete emission factor for an HMM using reduced data from
        observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`PseudocountEmissionEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        list
            list of :class:`~minihmm.factors.ArrayFactor` objects representing
            emission probabilities for each state
        """
        E = sum(reduced_data)
        E_sum = E.sum()
        E += get_model_noise(E, noise_weight, assymetric_weights=self.pseudocount_array)
        E += (pseudocount_weight * E_sum * self.pseudocount_array) / self.pseudocount_array.sum()
        E_normed = (E.T / E.sum(1)).T
        emission_factors = []
        for i in range(E_normed.shape[0]):
            emission_factors.append(ArrayFactor(E_normed[i, :]))

        return emission_factors


class TiedStatePriorEstimator(PseudocountStatePriorEstimator):
    """Estimate state prior probabilities, but tying (pooling data for and then
    jointly estimating) comparable states (e.g. all single states, all
    compound states).
    """

    def __init__(self, pseudocount_array, index_map):
        """
        pseudocount_array : float or numpy.ndarray
            Scalar (if evenly applying pseudocounts) or numpy array (if
            assymetrically weighting pseudocounts) pseudocounts to apply during
            estimation.

        index_map : numpy.ndarray
            a NUM_STATES-vector in which cells containing identical
            integer values have tied parameters.
        """
        self.index_map = copy.deepcopy(index_map)
        self.index_weights = numpy.array([(self.index_map == X).sum() \
                                          for X in range(0, 1+self.index_map.max())]
                                         )
        self.pseudocount_mask = (pseudocount_array > 0)
        PseudocountStatePriorEstimator.__init__(self, pseudocount_array)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct state prior factor for an HMM using reduced data from
        observation sequences

        Parameters
        ----------
        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`TiedStatePriorEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)


        Returns
        -------
        :class:`~minihmm.factors.ArrayFactor`
           Tied state priors
        """
        pi_raw = sum(reduced_data)
        pi_sum = pi_raw.sum()
        reduced_vector = numpy.zeros(1 + max(self.index_map))

        for i, val in enumerate(pi_raw):
            reduced_vector[self.index_map[i]] += pi_raw[i]

        # add noise
        reduced_vector += get_model_noise(
            reduced_vector, noise_weight, assymetric_weights=self.index_weights
        )  # FIXME: THIS WILL ADD NOISE TO FORBIDDEN CELLS

        # divide each starting cell by number of destination cells
        reduced_vector /= self.index_weights

        # populate destination vector
        pi_proc = numpy.zeros_like(pi_raw, dtype=float)
        for i in range(len(pi_proc)):
            pi_proc[i] = reduced_vector[self.index_map[i]]

        # add pseudocounts
        pi_proc += (pseudocount_weight * pi_sum *
                    self.pseudocount_array) / self.pseudocount_array.sum()
        pi_proc *= self.pseudocount_mask  # re-zero forbidden cells that received noise during tying

        # normalize
        pi_proc /= pi_proc.sum()

        return ArrayFactor(pi_proc)


class TiedTransitionEstimator(PseudocountTransitionEstimator):
    """Estimate state prior probabilities, but tying (pooling data for and then
    jointly estimating) comparable states (e.g. all single states, all
    compound states).
    """

    def __init__(self, pseudocount_array, index_map):
        """
        pseudocount_array : float or numpy.ndarray
            Scalar (if evenly applying pseudocounts) or numpy array (if
            assymetrically weighting pseudocounts) pseudocounts to apply during
            estimation.

        index_map : numpy.ndarray
            a `NUM_STATES x NUM_STATES` table vector in which cells
            containing identical integer values have tied parameters.
        """
        self.index_map = copy.deepcopy(index_map)
        self.index_weights = numpy.array([(self.index_map == X).sum() \
                                          for X in range(0, 1 + self.index_map.max())]
                                         )
        self.pseudocount_mask = (pseudocount_array > 0)
        PseudocountTransitionEstimator.__init__(self, pseudocount_array)

    def construct_factors(self, model, reduced_data, noise_weight=0, pseudocount_weight=1e-10):
        """Construct transition factor for an HMM using reduced data from
        observation sequences

        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`TiedTransitionEstimator.reduce_data`

        noise_weight : float, optional
            weight of noise to add, relative to number of of observations (e.g.
            transition counts, state prior counts, emission counts, et c) in
            data set. (Default: 0)

        pseudocount_weight : float, optional
            weight of pseudocounts to add, relative to number of of
            observations (transition counts, state prior counts, emission
            counts, et c) in data set (Default: 1e-8)

        Returns
        -------
        :class:`~minihmm.factors.MatrixFactor`
            Tied state transition probability table
        """
        A_raw = sum(reduced_data)
        A_sum = A_raw.sum()
        reduced_vector = numpy.zeros(1 + self.index_map.max())

        for i in range(A_raw.shape[0]):
            for j in range(A_raw.shape[1]):
                reduced_vector[self.index_map[i, j]] += A_raw[i, j]

        # add noise
        reduced_vector += get_model_noise(
            reduced_vector, noise_weight, assymetric_weights=self.index_weights
        )  # FIXME: THIS WILL ADD NOISE TO FORBIDDEN CELLS

        # divide each starting cell by number of destination cells
        reduced_vector /= self.index_weights

        # populate destination vector
        A_proc = numpy.zeros_like(A_raw, dtype=float)
        for i in range(A_proc.shape[0]):
            for j in range(A_proc.shape[1]):
                A_proc[i, j] = reduced_vector[self.index_map[i, j]]

        # add pseudocounts
        A_proc += (pseudocount_weight * A_sum *
                   self.pseudocount_array) / self.pseudocount_array.sum()
        A_proc *= self.pseudocount_mask  # re-zero forbidden cells that became zero via noise addition

        # normalize
        A_proc = (A_proc.T / A_proc.sum(1)).T

        return MatrixFactor(A_proc)


# TODO: implement model noise? by adding Gaussian noise around the observation vectors?
#        what would pseudocounts/regularization mean, if anything, in this context?
class UnivariateGaussianEmissionEstimator(AbstractProbabilityEstimator):
    """Estimate univariate Gaussian emissions from observation sequences in
    Baum-Welch training.
    """

    def is_valid(self, reduced_data):
        return not (numpy.isnan(reduced_data) | numpy.isinf(reduced_data)).any()

    def reduce_data(self, my_obs, obs_logprob, forward, backward, scale_factors, ksi):
        """Collect data from a single observation sequence and reduce it to a
        form amenable for factor construction

        Parameters
        ----------
        my_obs : list-like
            Observation sequence

        obs_logprob : float
            Observation logprob, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        forward : numpy.ndarray
            Scaled forward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        backward : numpy.ndarray
            Scaled backward probability matrix, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        scale_factors : numpy.ndarray
            Scale factors used in scaling `forward` and `backward` to
            floating-point friendly sizes, from
            :meth:`~minihmm.hmm.FirstOrderHMM.forward_backward`

        ksi : numpy.ndarray
            `MxNxT` array describing the full probability of being in state `M`
            at time `t` and state `N` at time `t+1`. From

        Returns
        -------
        numpy.ndarray
            Emission matrix for given observation sequence, in which each row
            is a model state. The first column is the estimated mean, the
            second the estimated variance, and the third, the number of points
            counted in each sequence
        """
        num_states = ksi.shape[1]
        my_E = numpy.zeros((num_states, 3))
        postprob = forward * backward
        tmp = postprob * scale_factors[:, None]
        for i in range(num_states):
            # n counted in mean
            num_observations = tmp[:, i].sum()
            my_E[i, 2] = num_observations

            # probability-weighted mean
            my_E[i, 0] = (tmp[:, i] * my_obs).sum() / num_observations

            # probability-weighted unbiased variance
            my_E[i, 1] = (tmp[:, i] * (my_obs - my_E[i, 0])**2).sum() / (num_observations - 1)

        return my_E

    def construct_factors(self, model, reduced_data, **ignored):
        """Construct discrete emission factor for an HMM using reduced data from
        observation sequences.

        Pooled variance estimation formula drawn from
        en.wikipedia.org/wiki/Pooled_variance

        Parameters
        ----------
        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`UnivariateTEmissionEstimator.reduce_data`


        Returns
        -------
        list
            List of :class:`~minihmm.factors.ScipyDistributionFactor` objects
            representing Gaussian emission factors for each state
        """
        E = numpy.zeros_like(reduced_data[0])
        var_pool_numerators = numpy.zeros(E.shape[0])
        for my_E in reduced_data:
            E[:, 0] += my_E[:, 0] * my_E[:, 2]
            E[:, 2] += my_E[:, 2]
            var_pool_numerators += my_E[:, 1] * (my_E[:, 2] - 1)

        E[:, 0] /= E[:, 2]
        E[:, 1] = var_pool_numerators / (E[:, 2] - len(reduced_data))  #dof-1 var correction

        emission_factors = []
        for i in range(E.shape[0]):
            emission_factors.append(
                ScipyDistributionFactor(scipy.stats.norm, loc=E[i, 0], scale=E[i, 1]**0.5)
            )

        return emission_factors


# NOT TESTED
class UnivariateTEmissionEstimator(UnivariateGaussianEmissionEstimator):
    """Estimate univariate T-distributed emissions from observation sequences
    in Baum-Welch training.
    """

    def construct_factors(self, model, reduced_data, **ignored):
        """Construct discrete emission factor for an HMM using reduced data from
        observation sequences.

        Pooled variance estimation formula drawn from
        en.wikipedia.org/wiki/Pooled_variance

        Parameters
        ----------
        model : :class:`~minihmm.hmm.FirstOrderHMM` or subclass

        reduced_data : numpy.ndarray
            sufficient statistics for observations, from
            :meth:`UnivariateTEmissionEstimator.reduce_data`


        Returns
        -------
        list
            list of :class:`~minihmm.factors.ScipyDistributionFactor` objects
            representing emission factors for each state
        """
        E = numpy.zeros_like(reduced_data[0])
        n = numpy.zeros(E.shape[0])
        var_pool_numerators = numpy.zeros(E.shape[0])
        for my_E in reduced_data:
            E[:, 0] += my_E[:, 0] * my_E[:, 2]
            E[:, 2] += my_E[:, 2]
            var_pool_numerators += my_E[:, 1] * (my_E[:, 2] - 1)
            n += my_E[:, 2]

        E[:, 0] /= n
        E[:, 1] = var_pool_numerators / (n - len(reduced_data))

        emission_factors = []
        for i in range(E.shape[0]):
            emission_factors.append(
                ScipyDistributionFactor(
                    scipy.stats.t,
                    n[i] - 1,  # dof
                    loc=E[i, 0],
                    scale=E[i, 1]**0.5
                )
            )

        return emission_factors
