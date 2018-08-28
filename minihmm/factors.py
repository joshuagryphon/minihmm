#!/usr/bin/env python
"""Pythonic representations of Factors (probability distributions, or, more
generally, affinity functions), which may serve as components of probabilistic
models (e.g. transition or observation probabilities for a
:class:`~minihmm.hmm.FirstOrderHMM` or as models themselves. All probabilistic
models in this library descend from :class:`~minihmm.factors.AbstractFactor`

Factors may be:

    - continuous or discrete
    - univariate or multivariate in their emissions

Factors must be able to:

    - calculate a log probability of an observation via a ``logprob()`` method

    - calculate a probability of an obervation via a ``probability()`` method

    - calculate a log probability of a series of observations via
      a ``get_model_log_likelihood()`` method

    - offer serialization support via pickling (we use :mod:`jsonpickle`
      and, where necessary, implement ``__reduce__()`` methods to make sane
      JSON blobs)

In addition, factors may:
    - supply a ``generate(n)`` function, to generate `n` observations
      from their distribution.

"""
import functools
import copy
import multiprocessing
import numpy
import scipy.stats

from abc import abstractmethod

from minihmm.util import matrix_from_dict, matrix_to_dict

#===============================================================================
# Abstract classes
#===============================================================================


class AbstractFactor(object):
    """Abstract class for all probability distributions
    """

    def probability(self, *args, **kwargs):
        """Return the probability of a single observation

        Parameters
        ----------
        args : list
            List of arguments representing observations

        kwargs : dict
            Dict of arguments representing observations

        Returns
        -------
        float
            Probability of observation
        """
        return numpy.exp(self.logprob(*args, **kwargs))

    def logprob(self, *args, **kwargs):
        """Return the log probability of a single observation

        Parameters
        ----------
        args : list
            List of arguments representing observations

        kwargs : dict
            Dict of arguments representing observations

        Returns
        -------
        float
            Log probability of observation
        """
        return numpy.log(self.probability(*args, **kwargs))

    def get_model_log_likelihood(self, observations, processes=4):
        """Return the log likelihood of a sequence of observations

        Parameters
        ----------
        observations : list-like
            Sequence of observation

        Processes : int, optional
            Number of processes to use when computing likelihood
            (Default: 4)

        Returns
        -------
        float
            Log likelihood of all observations
        """
        if processes == 1:
            return sum((self.logprob(X) for X in observations))
        else:
            pool = multiprocessing.Pool(processes=processes)
            pool_results = pool.map(self.logprob, observations)
            pool.close()
            pool.join()
            return sum(pool_results)

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Sample a random value from the distribution"""

    @abstractmethod
    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by `self.get_row()`
        """

    @abstractmethod
    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`
        """


#===============================================================================
# Helpers for unpickling / reviving from jsonpickle
#===============================================================================


def _get_scipydistfactor_from_dict(dtmp):
    """Revive a :class:`ScipyDistributionFactor` from a dictionary

    Parameters
    ----------
    dtmp : dict
        Dictionary, created by :meth:`ScipyDistributionFactor._to_dict`

    Returns
    -------
    ScipyDistributionFactor
    """
    # code note: this function cannot be a static method of
    # ScipyDistributionFactor because it is called by its __reduce__()
    # function
    dist_class = getattr(scipy.stats.distributions, dtmp["dist_class"])
    return ScipyDistributionFactor(dist_class(), *dtmp["dist_args"], **dtmp["dist_kwargs"])


#===============================================================================
# Implementable classes
#===============================================================================


class ArrayFactor(AbstractFactor):
    """Univariate probability distribution constructed from a 1D list or array

    Attributes
    ----------
    data : numpy.ndarray
        Table of probabilities indexed by position


    See also
    --------
    MatrixFactor
    """

    def __init__(self, data):
        """Create an ArrayFactor

        Parameters
        ----------
        data : list-like
            Array of probabilities indexed by position
        """
        self.data = numpy.array(data)

    def __eq__(self, other):
        return (self.data == other.data).all()

    def __len__(self):
        return len(self.data)

    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by :meth:`ArrayFactor.get_row`
        """
        return [str(X) for X in range(len(self.data))]

    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`
        """
        return list(self.data)

    def probability(self, i):
        """Return probability in cell `i` of the array

        Parameters
        ----------
        i : int
            Index of requested probability
        """
        return self.data[i]

    def generate(self):
        """Generate a random sample from the distribution

        Returns
        --------
        int
            Sample generated
        """
        return (self.data.cumsum() >= numpy.random.random()).argmax()


class MatrixFactor(AbstractFactor):
    """Bivariate probability distribution constructed from a two-dimensional
    matrix or array. MatrixFactors can represent joint distributions `P(X, Y)`
    as well as conditional distributions `P(Y|X)`, depending upon whether
    `self.row_conditional` is ``True`` or ``False``.

    Attributes
    ----------
    data : numpy.ndarray
        MxN table of probabilities

    row_conditional : bool, optional
        If ``True``, `data` is a conditional probability table,
        with rows specifying the conditional variable `P(column|row)`
        If ``False``, `data` is athejoint distribution of probabilities
        `P(column, row)`. (Default: ``True``)

    """

    def __init__(self, data, row_conditional=True):
        """Create a MatrixFactor

        Parameters
        ----------
        data : numpy.ndarray
            MxN table of probabilities

        row_conditional : bool, optional
            If ``True``, `data` is a conditional probability table, with rows
            specifying the conditional variable `P(column|row)` If ``False``,
            ``data`` is the joint distribution of probabilities
            `P(column,row)`. (Default: ``True``)
        """
        self.row_conditional = row_conditional
        self.data = numpy.array(data)

    def __eq__(self, other):
        return self.row_conditional == other.row_conditional and (self.data == other.data).all()

    def __len__(self):
        if self.row_conditional is True:
            return self.data.shape[0]
        else:
            return self.data.shape[0] * self.data.shape[1]

    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by :meth:`MatrixFactor.get_row`
        """
        shape = self.data.shape
        return ["%d,%d" % (X, Y) for X in range(shape[0]) for Y in range(shape[1])]

    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`
        """
        return list(self.data.ravel())

    def probability(self, i, j):
        """Return probability value at `(i, j)` in underlying matrix

        Parameters
        ----------
        i : int
            Row index

        j : int
            Column index

        Returns
        -------
        float
            `P(i|j)` if `self.row_conditional` is ``True``, or
            `P(i,j)` if `self.row_conditional` is ``False``.
        """
        return self.data[i, j]

    def generate(self, i=None, size=1):
        """Sample from the :class:`~minihmm.factors.MatrixFactor`

        Parameters
        ----------
        i : int or None
            Row index, required if `self.row_conditional` is ``True``,
            otherwise ignored

        Returns
        -------
        (int,int)
            Sample, as *(row,col)*
        """
        if self.row_conditional is True:
            #return (self.data[i,:].cumsum() >= numpy.random.random()).argmax()
            return (numpy.random.random() < self.data[i, :].cumsum()).argmax()
        else:
            #TODO: implement!
            pass


class FunctionFactor(AbstractFactor):
    """Probability distribution constructed from functions

    Attributes
    ----------
    _func : function
        Function that evaluates probability

    _generator : function
        Function that samples from the probability distribution

    func_args : list
        Zero or more positional arguments passed to `self._func`
        and `self._generator`

    func_kwargs : dict
        Zero or more keyword arguments passed to `self._func`
        and `self._generator`
    """

    def __init__(self, func, generator_func, *func_args, **func_kwargs):
        """Create a FunctionFactor

        Parameters
        ----------
        func : function
            Function that evaluates probability, given `func_args`,
            `func_kwargs`, and an observation

        generator_func : function
            Function that generates samples, given `func_args`
            and `func_kwargs`

        func_args
            Zero or more positional arguments to pass to `func`
            and `generator func`

        func_kwargs
            Zero or more keyword arguments to pass to `func`
            and `geneator func`

        """
        self.probability = functools.partial(func, *func_args, **func_kwargs)
        self._generator = functools.partial(generator_func, *func_args, **func_kwargs)
        self._func = func
        self._funcname = getattr(self._func, "func_name", getattr(self._func, "__name__", "foo"))
        self._generator_func = generator_func
        self.func_args = copy.deepcopy(func_args)
        self.func_kwargs = copy.deepcopy(func_kwargs)

    def __eq__(self, other):
        return self._func == other._func \
            and self.func_args == other.func_args \
            and self.func_kwargs == other.func_kwargs

    def __repr__(self):
        return "<%s func:'%s'>" % (self.__class__.__name__, self._funcname)

    def __str__(self):
        return repr(self)

    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by :meth:`FunctionFactor.get_row`
        """
        ltmp = [str(X) for X in range(len(self.func_args))]
        ltmp += sorted(self.func_kwargs.keys())
        return ltmp

    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`
        """
        ltmp = list(self.func_args)
        ltmp += [X[1] for X in sorted(self.func_kwargs.items())]
        return ltmp

    def generate(self, *args, **kwargs):
        """Sample a random value from the distribution

        args
            zero or more positional arguments to pass to pdf or pmf
            *in addition* to those in `self.func_args`

        kwargs
            zero or more keyword arguments to pass to pdf or pmf
            *in addition* to those in `self.func_kwargs`

        Returns
        -------
        object, type depending upon function
            A sample from the distribution
        """
        return self._generator(*args, **kwargs)


class LogFunctionFactor(FunctionFactor):
    """Probability distribution constructed from functions"""

    def __init__(self, func, generator_func, *func_args, **func_kwargs):
        """Create a LogFunctionFactor

        Parameters
        ----------
        func : function
            Function that evaluates log probability, given `func_args`,
            `func_kwargs`, and an observation

        generator_func : function
            Function that generates samples, given `func_args`
            and `func_kwargs`

        func_args
            Zero or more positional arguments to pass to `func` and
            `generator func`

        func_kwargs
            Zero or more keyword arguments to pass to `func` and
            `generator func`

        """
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self._func = func
        self._funcname = getattr(self._func, "func_name", getattr(self._func, "__name__", "foo"))
        self._generator_func = generator_func
        self.logprob = functools.partial(func, *func_args, **func_kwargs)
        self._generator = functools.partial(generator_func, *func_args, **func_kwargs)


class ScipyDistributionFactor(AbstractFactor):
    """Probability distribution using distributions provided by :mod:`scipy.stats`

    Attributes
    ----------
    distribution : class
        A class describing a distribution from :mod:`scipy.stats`

    Notes
    -----
    These cannot be used in parallelized processes because distributions
    in :mod:`scipy.stats` are not picklable!
    """
    from_dict = staticmethod(_get_scipydistfactor_from_dict)

    def __init__(self, dist_class, *dist_args, **dist_kwargs):
        """Create a ScipyDistributionFactor

        Parameters
        ----------
        dist_class : class
            class for a distribution from :mod:`scipy.stats`
            (e.g. :obj:`scipy.stats.poisson`)

        dist_args
            Zero or more positional arguments to pass to `dist_class`

        dist_kwargs
            Zero or mor keyword arguments to pass to `dist_class`
        """
        self.distribution = dist_class(*dist_args, **dist_kwargs)
        self._dist_class = dist_class
        self.dist_args = dist_args
        self.dist_kwargs = dist_kwargs

        self.prob_fn = self.distribution.pdf
        self.log_prob_fn = self.distribution.logpdf

        # determine whether distribution is continuous
        # or discrete
        try:
            self.distribution.pdf(3)
        except AttributeError:
            self.prob_fn = self.distribution.pmf
            self.log_prob_fn = self.distribution.logpmf

    def __eq__(self, other):
        return self._to_dict() == other._to_dict()

    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by :meth:`ScipyDistributionFactor.get_row`

        Returns
        -------
        list
            Names of parameters
        """
        ltmp = [str(X) for X in range(len(self.dist_args))]
        ltmp += sorted(self.dist_kwargs.keys())
        return ltmp

    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`

        Returns
        -------
        list
            Values of parameters
        """
        ltmp = list(self.dist_args)
        ltmp += [X[1] for X in sorted(self.dist_kwargs.items())]
        return ltmp

    def _to_dict(self):
        """Convenience function to define pickling/unpickling protocol
        
        Returns
        -------
        dict
            Dictionary of essential information to pickle
        """
        return {
            "model_class": "minihmm.factors.ScipyDistributionFactor",
            "dist_class": self._dist_class.__class__.__name__,
            "dist_args": list(self.dist_args),
            "dist_kwargs": dict(self.dist_kwargs),
        }

    def __reduce__(self):
        return _get_scipydistfactor_from_dict, (self._to_dict(), )

    def __repr__(self):
        return "<%s model:%s parameters:%s>" % (self.__class__.__name__,
                                                 self._dist_class,
                                                 ",".join([str(X) for X in self.dist_args])+\
                                                 ",".join(["%s:%s" % (K, V) for K, V in self.dist_kwargs.items()]) )

    def logprob(self, *args, **kwargs):
        """Return the log probability of observing an observation

        Parameters
        ----------
        args
            zero or more positional arguments to pass to logpdf or logpmf
            *in addition* to those in `self.dist_args`

        kwargs
            zero or more keyword arguments to pass to logpdf or logpmf
            *in addition* to those in `self.dist_kwargs`

        Returns
        -------
        float
            Log probability of observation
        """
        return self.log_prob_fn(*args, **kwargs)

    def probability(self, *args, **kwargs):
        """Return the probability of observing an observation

        args
            zero or more positional arguments to pass to pdf or pmf
            *in addition* to those in `self.dist_args`

        kwargs
            zero or more keyword arguments to pass to pdf or pmf
            *in addition* to those in `self.dist_kwargs`

        Returns
        -------
        float
            Probability of observation
        """
        return self.prob_fn(*args)

    def generate(self, *args, **kwargs):
        """Sample a random value from the distribution"""
        return self.distribution.rvs(*args, **kwargs)
