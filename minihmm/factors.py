#!/usr/bin/env python
"""Pythonic representations of Factors (probability distributions, or, more
generally, affinity functions), which may serve as components of probabilistic
models (e.g. transition or observation probabilities for a |FirstOrderHMM|)
or as models themselves. All probabilistic models in this library
descend from |AbstractFactor|.

Factors may be:
    - continuous (e.g. |NormalFactor| ) or discete (e.g. |ArrayFactor|)
    - univariate or multivariate

Factors must be able to:
    - calculate a log probability of an observation via a ``logprob()`` method
    - calculate a log probability of a series of observations via
      a ``get_model_log_likelihood()`` method
    - serialize and deserialize their parameters via ``serialize()`` and
      ``deserialize()`` methods, respectively. At present, we are taking 
      suggestions for a formal spec on what the serialization protocol should
      be. Right now, it is a free-for-all.

In addition, factors may:
    - supply a ``generate(n)`` function, to generate *n* observations
      from their distribution.

"""
import numpy
import functools
import copy
import multiprocessing
from abc import abstractmethod


class AbstractFactor(object):
    """Abstract class for all probability distributions
    """
    def __init__(self,*args,**kwargs):
        self._args   = args
        self._kwargs = kwargs
        self.data  = list(args)
        self._param_names = [""]*len(args)
        self._pos_args    = len(args)
        
        for k,v in sorted(kwargs.items()):
            self.data.append(v)
            self._param_names.append(k) 

    def serialize(self):
        """Return parameters in some useful, serialized format
        e.g. for reporting during successive rounds of training
        
        Returns
        -------
        str
            Serialized parameters


        See also
        --------
        deserialize
        """
        return "\t".join([str(X) for X in self.data])
    
    def deserialize(self,parameters):
        """Return a Factor of the same class, using new parameters

        Parameters
        ----------
        parameters : str
            Parameters, formatted by :py:meth:`serialize`


        Returns
        -------
        AbstractFactor
            Factor of same type, deserialized


        See Also
        --------
        serialize
        """
        data = parameters.strip("\n").split("\t")
        return self.__class__(*data)

    def probability(self,*args,**kwargs):
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
        return numpy.exp(self.logprob(*args,**kwargs))
    
    def logprob(self,*args,**kwargs):
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
        return numpy.log(self.probability(*args,**kwargs))
    
    def get_model_log_likelihood(self,observations,processes=4):
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
            pool_results = pool.map(self.logprob,observations)
            pool.close()
            pool.join()
            return sum(pool_results)
        

class AbstractGenerativeFactor(AbstractFactor):
    """A factor which is also generative. This must implement a :meth:`generate` function        
    """
    @abstractmethod
    def generate(self,*args,**kwargs):
        """Sample a random value from the distribution 
        """
        pass        

class AbstractTableFactor(AbstractGenerativeFactor):
    """Base class for probability distributions that are internally represented
    as tables. These typically will have multiple free parameters that must
    share group constraints (e.g. they must all sum to less than one)
    """

    def __repr__(self):
        return "<%s parameters:%s>" % (self.__class__.__name__,
                                       ",".join([str(X) for X in self.data]))
    
    def __str__(self):
        return str(self.data)

    @abstractmethod
    def get_parameter_group_indices(self):
        """Return indices that indicate which parameters in the array
        returned by self.get_free_parameters() must sum to one or less

        Returns
        -------
        list
            list of *(int,int)* of starting and end half-open end positions
            for each group of parameters
        """
        pass
    

class ArrayFactor(AbstractTableFactor):
    """Univariate probability distribution constructed from a 1D list or array

    Attributes
    ----------
    data : numpy.ndarray
        Table of probabilities indexed by position


    See also
    --------
    MatrixFactor
    """
    def __init__(self,data):
        """Create an ArrayFactor

        Parameters
        ----------
        data : list-like
            Array of probabilities indexed by position
        """
        self.data = copy.deepcopy(numpy.array(data))
    
    def __len__(self):
        return len(self.data)
    
    def probability(self,i):
        """Return probability in cell ``i`` of the array
       
        Parameters
        ----------
        i : int
            Index of requested probability   
        """
        return self.data[i]

    def serialize(self):
        """Return free parameters in the model (i.e. all but last cell),
        as a tab-delimited string.
        
        This includes all but the final element in self.data, since
        these must sum to 1.0

        Returns
        -------
        str
            String representation of parameters

        See also
        --------
        ArrayFactor.deserialize
        """
        return "\t".join([str(X) for X in self.data[:-1]])
    
    def get_parameter_group_indices(self):
        """Return indices that indicate which free parameters in the array
        returned by self.serialize() must sum to one or less

        @return list<(int,int)> of starting and half-open end positions
                                of each group.
        """
        return [(0,len(self)-1)]
            
    def generate(self):
        """Generate a random sample from the distribution
        
        @return Sample
        """
        return (self.data.cumsum() >= numpy.random.random()).argmax()
    
    def deserialize(self,paramstr):
        """Return a new ArrayFactor, using updated parameters
        
        Parameters
        ----------
        paramstr : str
            Tab-delimited string of free parameters in the model
            (i.e. excluding last cell).

        Returns
        -------
        |ArrayFactor|
            ArrayFactor with probabilities specified in ``paramstr``.            

        See Also
        --------
        ArrayFactor.serialize
        """
        parameters = [float(X) for X in paramstr.strip("\n").split("\t")]
        assert len(parameters) == len(self.serialize())
        new_parameters = list(parameters)
        new_parameters.append(1.0 - sum(parameters))
        return ArrayFactor(numpy.array(new_parameters))


class MatrixFactor(AbstractTableFactor):
    """Bivariate probability distribution constructed from a two-dimensional
    matrix or array. MatrixFactors can represent joint distributions *P(X,Y)*
    as well as conditional distributions *P(Y|X)*, depending upon whether
    ``self.row_conditional`` is *True* or *False*. If *True*, the |MatrixFactor|
    is assumed to represent a conditional distribution, and outputs from
    :meth:`generate`, :meth:`serialize`, and :meth:`deserialize` all take this
    into account.

    Attributes
    ----------
    data : numpy.ndarray
        MxN table of probabilities

    row_conditional : bool, optional
        If *True*, ``data`` is a conditional probability table,
        with rows specifying the conditional variable *P(column|row)*
        If *False*, ``data`` is athejoint distribution of probabilities
        *P(column,row)*. (Default: *True*)

    """
    def __init__(self,data,row_conditional=True):
        """Create a MatrixFactor
        
        Parameters
        ----------
        data : numpy.ndarray
            MxN table of probabilities

        row_conditional : bool, optional
            If *True*, ``data`` is a conditional probability table,
            with rows specifying the conditional variable *P(column|row)*
            If *False*, ``data`` is athejoint distribution of probabilities
            *P(column,row)*. (Default: *True*)
        """
        self.row_conditional = row_conditional
        self.data = numpy.array(copy.deepcopy(data))
    
    def __len__(self):
        if self.row_conditional is True:
            return self.data.shape[0]
        else:
            return self.data.shape[0]*self.data.shape[1]
            
    def probability(self,i,j):
        """Return probability value at *(i,j)* in underlying matrix
        
        Parameters
        ----------
        i : int
            Row index

        j : int
            Column index

        Returns
        -------
        float
            *P(i|j)* if ``self.row_conditional`` is *True*, or
            *P(i,j)* if ``self.row_conditional`` is *False*.
        """
        return self.data[i,j]
    
    def generate(self,i=None,size=1):
        """Sample from the |MatrixFactor|
        
        Parameters
        ----------
        i : int or None
            Row index, required if ``self.row_conditional`` is *True*,
            otherwise ignored

        Returns
        -------
        (int,int)
            Sample, as *(row,col)*
        """
        if self.row_conditional is True:
            #return (self.data[i,:].cumsum() >= numpy.random.random()).argmax()
            return (numpy.random.random() < self.data[i,:].cumsum()).argmax()
        else:
            #TODO: implement!
            pass

    def get_parameter_group_indices(self):
        """Return indices that indicate which free parameters in the array
        returned by self.serialize() must sum to one or less
       
        Returns
        -------
        list
            list of tuples of *(int,int)* of starting and end half-open
            end positions of each group.
        """
        if self.row_conditional is True:
            ltmp = []
            for i in range(1,len(self)+1):
                start = (i-1)*(self.shape[1]-1)
                end   = i*(self.shape[1]-1)
                ltmp.append((start,end))
            return ltmp
        else:
            return [(0,len(self)-1)]
    
    def serialize(self):
        """Return **free parameters** in the model as a tab-delimited string.
        
        If ``self.row_conditional`` is *True*, this includes all but the final element
        in each row in self.data, since each row must sum to 1.0
        
        If ``self.row_conditional`` is *False*, this includes all but the final cell
        in ``self.data``, since in thise case the entire matrix must sum to 1.0


        Returns
        -------
        str
            tab-delimited string of parameters in model

        See also
        --------
        MatrixFactor.deserialize
        """
        if self.row_conditional is True:
            return "\t".join([str(X) for X in self.data[:,0:-1].ravel()])
        else:
            return "\t".join([str(X) for X in self.data.ravel()[:-1]])
    
    def deserialize(self,param_str):
        """Returns a MatrixFactor using updated parameters and the same
        row conditionality as this MatrixFactor

        Parameters
        ----------
        param_str
            A tab-delimited string of **free parameters**, as formatted by
            :meth:`MatrixFactor.serialize`

        Returns
        -------
        |MatrixFactor|
            |MatrixFactor| constructed from parameters

        See also
        --------
        MatrixFactor.serialize
        """
        parameters = [float(X) for X in param_str.strip("\n").split("\t")]
        assert len(parameters) == len(self.serialize())
        
        if self.row_conditional is True:
            new_matrix = numpy.zeros(self.data.shape)
            tmp = numpy.matrix(parameters).reshape(self.data.shape[0],self.data.shape[1]-1)
            new_matrix[:,0:-1] = tmp
            new_matrix[:,-1]   = 1.0 - new_matrix.sum(1)
        else:
            parameters = list(parameters)
            parameters.append(1.0-sum(parameters))
            new_matrix = numpy.matrix(parameters).reshape(self.data.shape)
         
        return MatrixFactor(new_matrix,row_conditional=self.row_conditional)


class FunctionFactor(AbstractGenerativeFactor):
    """Probability distribution constructed from functions

    Attributes
    ----------
    _func : function
        Function that evaluates probability

    _generator : function
        Function that samples from the probability distribution

    func_args : list
        Zero or more positional arguments passed to ``self._func``
        and ``self._generator``

    func_kwargs : dict
        Zero or more keyword arguments passed to ``self._func``
        and ``self._generator``
    """
    def __init__(self,func,generator_func,*func_args,**func_kwargs):
        """Create a FunctionFactor
        
        Parameters
        ----------
        func : function
            Function that evaluates probability, given ``func_args``,
            ``func_kwargs``, and an observation
 
        generator_func : function
            Function that generates samples, given ``func_args``
            and ``func_kwargs``

        func_args
            Zero or more positional arguments to pass to ``func``
            and ``generator func``

        func_kwargs
            Zero or more keyword arguments to pass to ``func``
            and ``geneator func``

        """
        self.probability  = functools.partial(func,*func_args,**func_kwargs)
        self._generator   = functools.partial(generator_func,*func_args,**func_kwargs)
        self._func        = func
        self._generator_func = generator_func
        self.func_args   = copy.deepcopy(func_args)
        self.func_kwargs = copy.deepcopy(func_kwargs)
 
    def __repr__(self):
        return "<%s func:%s() parameters:%s>" % (self.__class__.__name__,
                                                 self._func.func_name,
                                                 ",".join([str(X) for X in self.data]))
    
    def __str__(self):
        return "(%s,%s)" % (self._func.func_name,",".join([str(X) for X in self.data]))
        
    def serialize(self):
        """Serialize |FunctionFactor|, saving ``self.func_args`` and
        ``self.func_kwargs`` to a tab-delimited string.

        See also
        --------
        FunctionFactor.deserialize

        Notes
        -----
        Serialization/deserialization formats should NOT be considered stable
        """
        outp   = "\t".join([str(X) for X in self.func_args])
        outp  += "\t".join(["%s=%s" % (K,V) for K,V in sorted(self.func_kwargs.items())])
        return outp

    def deserialize(self,param_str):
        """Create a |FunctionFactor| using the same base functions, but with
        new parameters

        Parameters
        ----------
        param_str : str
            Tab-delimited string of function and keyword arguments

        Returns
        -------
        |FunctionFactor|

        See also
        --------
        FunctionFactor.serialize
        """
        args   = []
        kwargs = {}
        items  = param_str.strip("\n").split("\t")

        for item in items:
            if "=" not in item:
                args.append(guess_formatter(item))
            else:
                key,val = item.split("=")
                kwargs[key] = guess_formatter(val)

        assert len(args)   == len(self.func_args)
        assert len(kwargs) == len(self.func_kwargs)
        return self.__class__(self._func,
                              self._generator_func,
                              *args,**kwargs)
    
    def generate(self,*args,**kwargs):
        """Sample a random value from the distribution 
       
        args
            zero or more positional arguments to pass to pdf or pmf
            **in addition** to those in ``self.func_args``
        
        kwargs
            zero or more keyword arguments to pass to pdf or pmf
            **in addition** to those in ``self.func_kwargs``

        Returns
        -------
        object, type depending upon function
            A sample from the distribution
        """
        return self._generator(*args,**kwargs)


class LogFunctionFactor(FunctionFactor):
    """Probability distribution constructed from functions
    """
    def __init__(self,func,generator_func,func_args=[]):
        """Create a LogFunctionFactor
        
        Parameters
        ----------
        func : function
            Function that evaluates log probability, given ``func_args``,
            ``func_kwargs``, and an observation
 
        generator_func : function
            Function that generates samples, given ``func_args``
            and ``func_kwargs``

        func_args
            Zero or more positional arguments to pass to ``func``
            and ``generator func``

        func_kwargs
            Zero or more keyword arguments to pass to ``func``
            and ``geneator func``

        """
        self.func_args   = func_args
        self.func_kwargs = func_kwargs
        self._func           = func
        self._generator_func = generator_func
        self.logprob    = functools.partial(func,*func_args,**func_kwargs)
        self._generator = functools.partial(generator_func,*func_args,**func_kwargs)


class ScipyDistributionFactor(AbstractGenerativeFactor):
    """Probability distribution using distributions provided by :mod:`scipy.stats`

    Attributes
    ----------
    distribution : class
        A class describing a distribution from :mod:`scipy.stats`
    
    Notes
    -----
    These cannot be used in parallelized processes because distributions
    in :py:mod:`scipy.stats` are not picklable!
    """
    def __init__(self,dist_class,*dist_args,**dist_kwargs):
        """Create a ScipyDistributionFactor
        
        Parameters
        ----------
        dist_class : class
            class for a distribution from :mod:`scipy.stats`
            (e.g. :obj:`scipy.stats.poisson`)
        
        dist_args
            Zero or more positional arguments to pass to ``dist_class``

        dist_kwargs
            Zero or mor keyword arguments to pass to ``dist_class``
        """
        self.distribution = dist_class(*dist_args,**dist_kwargs)
        self._dist_class  = dist_class
        self.dist_args   = dist_args
        self.dist_kwargs = dist_kwargs
        
        self.prob_fn     = self.distribution.pdf
        self.log_prob_fn = self.distribution.logpdf

        # determine whether distribution is continuous 
        # or discrete
        try:
            self.distribution.pdf(3)
        except AttributeError:
            self.prob_fn = self.distribution.pmf
            self.log_prob_fn = self.distribution.logpmf
        #AbstractFactor.__init__(self,*dist_args,**dist_kwargs)

    def __repr__(self):
        return "<%s model:%s parameters:%s>" % (self.__class__.__name__,
                                                 self._dist_class,
                                                 ",".join([str(X) for X in self._args])+\
                                                 ",".join(["%s:%s" % (K,V) for K,V in self._kwargs.items()]) )

    def serialize(self):
        """Serialize |FunctionFactor|, saving ``self.func_args`` and
        ``self.func_kwargs`` to a tab-delimited string.

        See also
        --------
        ScipyDistributionFactor.deserialize

        Notes
        -----
        Serialization/deserialization formats should NOT be considered stable
        """
        outp   = "\t".join([str(X) for X in self.dist_args])
        outp  += "\t".join(["%s=%s" % (K,V) for K,V in sorted(self.dist_kwargs.items())])
        return outp

    def deserialize(self,param_str):
        """Create a new |ScipyDistributionFactor| from parameters

        Parameters
        ----------
        param_str : str
            Parameters, formatted as by :meth:`serialize`

        Notes
        -----
        serialization/deserialization parameters should NOT be considered stable

        See also
        --------
        ScipyDistributionFactor.serialize
        """
        new_args, new_kwargs = self._parse_new_parameters(parameters)
        return ScipyDistributionFactor(self._dist_class,*new_args,**new_kwargs)
        
    def logprob(self,*args,**kwargs):
        """Return the log probability of observing an observation

        Parameters
        ----------
        args
            zero or more positional arguments to pass to logpdf or logpmf
            **in addition** to those in ``self.dist_args``
        
        kwargs
            zero or more keyword arguments to pass to logpdf or logpmf
            **in addition** to those in ``self.dist_kwargs``

        Returns
        -------
        float
            Log-probability of observation
        """
        return self.log_prob_fn(*args,**kwargs)
    
    def probability(self,*args,**kwargs):
        """Return the probability of observing an observation
        
        args
            zero or more positional arguments to pass to pdf or pmf
            **in addition** to those in ``self.dist_args``
        
        kwargs
            zero or more keyword arguments to pass to pdf or pmf
            **in addition** to those in ``self.dist_kwargs``

        Returns
        -------
        float
            Probability of observation
        """
        return self.prob_fn(*args)
    
    def generate(self,*args,**kwargs):
        """Sample a random value from the distribution 
        """
        return self.distribution.rvs(*args,**kwargs)
