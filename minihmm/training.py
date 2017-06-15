#!/usr/bin/env python
"""A parallelized, extensible implementation of the Baum-Welch training algorithm
built atop plugin estimator classes, which determine how parameters for the HMM
are re-estimated from observation data during training (i.e. the E and M steps
of Expectation maximization). 

Also includes helper functions to add decaying noise during training
"""
from minihmm.hmm import FirstOrderHMM
from minihmm.factors import ArrayFactor, MatrixFactor, FunctionFactor, \
                            LogFunctionFactor, ScipyDistributionFactor
from minihmm.estimators import DiscreteStatePriorEstimator,\
                               DiscreteTransitionEstimator,\
                               DiscreteEmissionEstimator
from minihmm.util import NullWriter
import numpy
import datetime
import multiprocessing
import functools
import itertools

# TODO: change to biufc
_number_types = { int, long, float, numpy.long, numpy.longlong, numpy.longdouble, numpy.longfloat, 
                  numpy.uint, numpy.uint0, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                  numpy.int,  numpy.int0,  numpy.int8,  numpy.int16,  numpy.int32,  numpy.int64,
                  numpy.float, numpy.float16, numpy.float32, numpy.float64, numpy.float128,
                  numpy.ulonglong,
                }


#===============================================================================
# INDEX: helper functions
#===============================================================================

def neg_exp_noise_gen(a=1.0, b=1.0, offset=0):
    """Generate exponentially decaying constants following y = ae**-bx
    
    Parameters
    ----------
    A : float, optional
        Initial (maximum) value (Default: 1.0)
    
    B : float, optional
        Decay constant (Default: 1.0)
    
    offset : int, optional
        Starting offset in time (in case of training restart)
    
    Yields
    ------
    float
        Next noise amount
    """
    offset -= 1
    while True:
        offset += 1
        yield a*numpy.exp(-offset*b)

def linear_noise_gen(m=-0.05, b=1, offset=0):
    """Generate linearly decaying noise following y = max(m*x + b,0)
    
    Parameters
    ----------
    m : float, optional
        Slope of line (default: -0.05)
        
    b : float, optional
        Y-intercept of line (Default: 1)

    offset : int, optional
        Starting offset in time (in case of training restart)
    
    Yields
    ------
    float
        Next noise amount
    """
    offset -= 1
    while True:
        offset += 1
        yield max(m*offset+b,0)


#===============================================================================
# INDEX: training functions
#===============================================================================

def bw_worker(my_model,
              my_tuple,
              state_prior_estimator = DiscreteStatePriorEstimator(),
              emission_estimator    = None,
              transition_estimator  = DiscreteTransitionEstimator()):
    """Collect summary statistics from an observation sequence for Baum-Welch training.
    In an expectation-maximization context, :py:func:`bw_worker` is used in
    evaluating the Q function in the E step.
    
    
    Parameters
    ----------
    my_model : |FirstOrderHMM|
        Model under which observations are evaluated
    
    my_tuple : tuple
        Tuple of (:class:`numpy.ndarray`, :class:`float`) corresponding to an observation
        sequence and its weight

    state_prior_estimator : instance of subclass of |AbstractProbabilityEstimator|, optional
        Estimator that extracts summary statistics regarding probabilities of
        starting in each state. Typically a |DiscretePseudocountStatePriorEstimator|
                                    
    transition_estimator : instance of subclass of |AbstractProbabilityEstimator|, optional
        Estimator that extracts summary statistics regarding transition probabilities.
        Typically a |DiscretePseudocountTransitionEstimator|
                                    
    emission_estimator : instance of subclass of |AbstractProbabilityEstimator|, optional
        Estimator that extracts summary statistics regarding observations
        for each state


    Returns
    -------
    float
        log probability for observation sequence under ``my_model``

    int
        Length of observation sequence
   
    float
        Weight of observation sequence

    tuple
        numpy.ndarray
            contribution of ``my_obs`` to transition summary statistics
        
        numpy.array
            contribution of ``my_obs`` to emission summary statistics
            
        numpy.array
            contribution of ``my_obs`` to state prior summary statistics
    """
    my_obs, my_weight = my_tuple
    obs_logprob, forward, backward, scale_factors, ksi = my_model.forward_backward(my_obs)
    my_A  = transition_estimator.reduce_data(my_obs,obs_logprob,forward,backward,scale_factors,ksi)
    my_E  = emission_estimator.reduce_data(my_obs,obs_logprob,forward,backward,scale_factors,ksi)
    my_pi = state_prior_estimator.reduce_data(my_obs,obs_logprob,forward,backward,scale_factors,ksi)

    return obs_logprob, len(my_obs), my_weight, (my_A, my_E, my_pi)


def format_for_logging(x, fmt="%.8f"):
    """Format a model parameter for logging in a text file.
    Numerical types are formatted following the 'fmt' parameter.
    Lists and Numpy arrays are formatted as comma-separated strings.
    Other types are printed using their str() representation.
    
    Parameters
    ----------
    x : object, float, int, str, et c
        parameter
    
    fmt : str, optional
        printf-style format used for numerical types only
        (default: "%.32e")
    
    
    Returns
    -------
    str
        Formatted parameter
    """
    if type(x) in _number_types:
        return fmt % x
    if isinstance(x,type([])):
        return ",".join([format_for_logging(X) for X in x])
    if isinstance(x,numpy.ndarray):
        return ",".join([format_for_logging(X) for X in x])
    else:
        return str(x)



def train_baum_welch2(model,
                      obs,
                      state_prior_estimator   = DiscreteStatePriorEstimator(),
                      transition_estimator    = DiscreteTransitionEstimator(),
                      emission_estimator      = None,
                      weights                 = None,
                      pseudocount_weights     = iter([1e-10]),
                      observation_weights     = None,
                      noise_weights           = iter([0.0]),
                      log_func                = None,
                      learning_threshold      = 1e-5,
                      miniter                 = 10,
                      maxiter                 = 1000,
                      start_iteration         = 0,
                      processes               = 4,
                      chunksize               = None,
                      logfile                 = NullWriter(),
                      printer                 = NullWriter(),
                      freeze                  = None,
                     ):


    expectation_func = functools.partial(bw_worker,
                                         model,
                                         state_prior_estimator = state_prior_estimator,
                                         emission_estimator    = emission_estimator,
                                         transition_estimator  = transition_estimator
                                        )        

    def validation_func(obs_stats):
        """
        Must return bool
        """
        return not any([transition_estimator.is_invalid(obs_stats[0]),
                        emission_estimator.is_invalid(obs_stats[1]),
                        state_prior_estimator.is_invalid(obs_stats[2]),
                       ])

    def maximization_func(obs_stats,
                          obs_weights,
                          last_model,
                          noise_weight        = 0,
                          pseudocount_weights = None,
                          pseudocount_model   = None,
                          freeze              = None):
        """
        Must return new model
        """
        if freeze is None:
            freeze = set()

        A_parts, E_Parts, pi_parts = zip(*obs_stats)

        if "state_priors" in freeze:
            new_state_priors = last_model.state_priors
        else:
            new_state_priors = state_prior_estimator.construct_factors(last_model,
                                                                       pi_parts,
                                                                       noise_weight,
                                                                       pseudocounts)

        if "trans_probs" in freeze:
            new_trans_probs = last_model.trans_probs
        else:
            new_trans_probs = transition_estimator.construct_factors(last_model,
                                                                     A_parts,
                                                                     noise_weight,
                                                                     pseudocounts)

        if "emission_probs" in freeze:
            new_emission_probs = last_model.emission_probs
        else:
            new_emission_probs = emission_estimator.construct_factors(model,
                                                                      E_parts,
                                                                      noise_weight,
                                                                      pseudocounts)

        return FirstOrderHMM(state_priors   = new_state_priors,
                             trans_probs    = new_trans_probs,
                             emission_probs = new_emission_probs)

    return train_em(model,
                    obs,
                    expectation_func  = expectation_func,
                    validation_func   = validation_func,
                    maximization_func = maximization_func,
                    log_func          = log_func,

                    pseudocount_model   = None,

                    pseudocount_weights = pseudocount_weights,
                    noise_weights       = noise_weights,
                    observation_weights = observtaion_weights,

                    freeze              = freeze,

                    learning_threshold  = learning_threshold,
                    miniter             = miniter,
                    maxiter             = maxiter,
                    start_iteration     = start_iteration,

                    processes           = processes,
                    chunksize           = chunksize)
 

def train_baum_welch(model,
                     obs,
                     state_prior_estimator   = DiscreteStatePriorEstimator(),
                     transition_estimator    = DiscreteTransitionEstimator(),
                     emission_estimator      = None,
                     pseudocount_weights     = 0.0,
                     noise_weights           = 0.0,
                     observation_weights     = None,
                     learning_threshold      = 1e-5,
                     miniter                 = 10,
                     maxiter                 = 1000,
                     start_iteration         = 0,
                     processes               = 4,
                     chunksize               = None,
                     logfile                 = NullWriter(),
                     printer                 = NullWriter(),
                    ):
    """Train an HMM using the Baum-Welch algorithm on one or more unlabeled observation sequences.
    
    After Durbin et al., incorporating equations from solution guide, which
    differ from those in Rabiner

    Parameters
    ----------
    model : |FirstOrderHMM| or subclass thereof
        Starting HMM, initialized with some set of initial parameters
    
    obs : list of numpy.ndarray s
        One or more observation sequences which will be used for training.

    state_prior_estimator : instance of any subclass of |AbstractStatePriorEstimator|, optional
        (Default: instance of |DiscreteStatePriorEstimator|)
                                    
    transition_estimator : instance of any subclass of |AbstractTransitionEstimator|, optional
        (Default: instance of |DiscreteTransitionEstimator|)
                                    
    emission_estimator : instance of any subclass of AbstractEmissionEstimator

    pseudocount_weights : iterable or number, optional
        Iterator/generator indicating the total weight of pseudocounts that
        should be applied relative to total number of observations in data
        in each cycle of training. How the pseudocounts are distributed 
        is up to the Estimators supplied above.
        
        If an iterator/generator and not infinite, *1e-12* pseudocounts
        will be added after the iterator/generator raises 
        StopIteration. (Default: 0)
    
    noise_weights : iterable or number, optional
        Iterator/generator specifying the weight of noise relative to total
        number of observations in the internal observation matrices
        to add to each cycle.  How the noise is distributed 
        is up to the Estimators supplied above.
        
        If an iterator/generator and not infinite, no noise will be added
        after the iterator/generator raises  StopIteration.
        (Default: 0 (no noise))

    miniter : int or 0, optional
        Minimum number of training cycles (e.g. during which end_delta is ignored).
        This is useful for allowing added noise to choose suboptimal parameters
        int he hopes of better exploring the parameter space. (Default: *100*)

    maxiter : int or numpy.inf, optional
        Maximum number of training cycles. Specify numpy.inf to disable
        this termination criterion. (Default: *1000*)
    
    learning_threshold : float, optional
        Minimum change in total log likelihood required to continue training.
        Specify -numpy.inf to disable this termination criterion (Default: *1e-5*)
    
    start_iteration : int, optional
        Label for starting iteration (in case training was halted and resumed)
    
    processes : int, optional
        Number of processes to use during training (default: *4*)

    chunksize : int, optional
        Number of observation sequences to send to each process at a time
        (Default: calculated from ``len(obs)`` and ``processes``)
                          
    printer : file-like
        An open filehandle or object implementing a ``write()`` method,
        to which parameter values and the current likelihood will be passed
        at each iteration of training (Default: *|NullWriter|*)
    
    Returns
    -------
    |FirstOrderHMM|
        Best (highest-likelihood) model from training trajectory

    |FirstOrderHMM|
        Final model from training trajectory

    str
        Reason explaining why training ceased. *MAXITER* if maximum iterations
        reached. *CONVERGENCE* if model converged to ``learning_threshold``.
        *REGRESSION* if new parameters are worse than old ones, perhaps
        due to noise, but the training iteration is past ``miniter``
    """
    if emission_estimator is None:
        raise AssertionError("Emission estimator required by train_baum_welch(), but not supplied")

    oopsie = False

    state_priors     = model.state_priors
    emission_factors = model.emission_probs
    trans_probs      = model.trans_probs
    model_type       = type(model)
    
    models   = []
    logprobs = []
    unweight_logprobs = []

    logprobs_per_length          = []
    unweight_logprobs_per_length = []

    last_total_logprob = -numpy.inf
    delta              = numpy.inf
    c                  = start_iteration

    if chunksize is None:
        chunksize = len(obs) / (4*processes)
    if chunksize < 1:
        chunksize = len(obs) / processes
    if chunksize < 1:
        chunksize = 1

    if isinstance(pseudocount_weights, (int, float)):
        pseudocount_weights = itertools.repeat(1.0*pseudocount_weights)
    if isinstance(noise_weights, (int, float)):
        noise_weights = itertools.repeat(1.0*noise_weights)

    if observation_weights is None:
        observation_weights = [1.0]*len(obs)
        
    weighted_obs = zip(obs, observation_weights)

    try:
        while c < maxiter and (delta > learning_threshold or c < miniter) and not oopsie:
            model = model_type(state_priors,emission_factors, trans_probs)
            try:
                noise_weight = noise_weights.next()
            except StopIteration:
                noise_weight = 0

            try:
                pseudocounts = pseudocount_weights.next()
            except StopIteration:
                pseudocounts = 1e-12


            # E-step of Expectation-Maximization
            training_func = functools.partial(bw_worker,
                                              model,
                                              state_prior_estimator=state_prior_estimator,
                                              emission_estimator=emission_estimator,
                                              transition_estimator=transition_estimator
                                              )        
            if processes == 1:
                pool_results = (training_func(X) for X in weighted_obs)
            else:

                pool = multiprocessing.Pool(processes=processes)
                pool_results = pool.map(training_func, weighted_obs, chunksize)
                pool.close()
                pool.join()

            results      = []
            anti_results = []
            for my_result in pool_results:
                if any([numpy.isnan(my_result[0]),
                        numpy.isinf(my_result[0]),
                        transition_estimator.is_invalid(my_result[-1][0]),
                        emission_estimator.is_invalid(my_result[-1][1]),
                        state_prior_estimator.is_invalid(my_result[-1][2]),
                        ]):
                    anti_results.append(my_result)
                else:
                    results.append(my_result)
            
            counted = len(results)
            if counted == 0:
                oopsie = True
                break

            obs_logprobs, obs_lengths, obs_weights, obs_stats = zip(*results)
            A_parts, E_parts, pi_parts = zip(*obs_stats)

            obs_weights  = numpy.array(obs_weights)
            obs_lengths  = numpy.array(obs_lengths)
            obs_logprobs = numpy.array(obs_logprobs)

            weight_logprobs = obs_weights * obs_logprobs
            weight_lengths  = obs_weights * obs_lengths

            new_total_logprob = weight_logprobs.sum()
            total_obs_length  = weight_lengths.sum()        
            logprob_per_obs = new_total_logprob / total_obs_length
            logprobs_per_length.append(logprob_per_obs)

            unweight_total_logprob = obs_logprobs.sum()
            unweight_total_length  = obs_lengths.sum()
            unweight_logprob_per_obs = unweight_total_logprob / unweight_total_length
            unweight_logprobs_per_length.append(unweight_logprob_per_obs)

            models.append(model)
            logprobs.append(new_total_logprob)
            unweight_logprobs.append(unweight_total_logprob)

            delta              = new_total_logprob - last_total_logprob
            last_total_logprob = new_total_logprob

           
            # record parameters & test convergence
    #         log_message = "%s\t%.10f\t%.6e\t%s\t%s\t%s" % (datetime.datetime.now(),
    #                                                  new_total_logprob,
    #                                                  logprob_per_obs,
    #                                                  counted,
    #                                                  c,
    #                                                  "\t".join([format_for_logging(X) for X in params])
    #                                                  )
            print_message = "%s\t%s\t%s\t%s\t%s\t%s" % (datetime.datetime.now(),
                                                 new_total_logprob,
                                                 logprob_per_obs,
                                                 counted,
                                                 c,
                                                 model.serialize()
    #                                             "\t".join([format_for_logging(X,fmt="%.4f") for X in params])
                                                 )
    #         logfile.write("%s\n" % log_message)
            print("%s\n" % print_message)
                

            # M-step of Expectation-Maximization:
            # Update parameters for next model
            state_priors = state_prior_estimator.construct_factors(model,
                                                                   pi_parts,
                                                                   noise_weight,
                                                                   pseudocounts)
            trans_probs = transition_estimator.construct_factors(model,
                                                                 A_parts,
                                                                 noise_weight,
                                                                 pseudocounts)
            emission_factors = emission_estimator.construct_factors(model,
                                                                    E_parts,
                                                                    noise_weight,
                                                                    pseudocounts)
            
            c += 1

    except KeyboardInterrupt:
        halted = True

        
    # report why training stopped
    if c == maxiter:
        reason = "MAXITER"
    elif delta <= learning_threshold:
        if delta >= 0:
            reason = "CONVERGENCE"
        else:
            reason = "REGRESSION"
    elif oopsie == True:
        reason = "ALL_INVALID"
    elif halted == True:
        reason = "HALTED"
    else:
        reason = "SOMETHING TERRIBLE: iter %s, delta %s, counted %s" % (c,delta,counted)
 
    logprobs   = numpy.array(logprobs)
    best_model = models[logprobs.argmax()]

    dtmp = {
        "best_model"                   : best_model,
        "final_model"                  : model,
        "weight_logprobs"              : logprobs,
        "unweight_logprobs"            : unweight_logprobs,
        "weight_logprobs_per_length"   : logprobs_per_length,
        "unweight_logprobs_per_length" : unweight_logprobs_per_length,
        "reason"                       : reason,
        "iterations"                   : c - 1,
        "models"                       : models,
    }
 
       
    return dtmp
