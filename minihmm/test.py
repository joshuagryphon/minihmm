#!/usr/bin/env python
"""Testing functions"""
import numpy
import sys
import functools
from minihmm.hmm import FirstOrderHMM
from minihmm.estimators import UnivariateGaussianEmissionEstimator,\
                                    DiscreteEmissionEstimator
from minihmm.training import train_baum_welch, neg_exp_noise_gen
from minihmm.factors import ArrayFactor, MatrixFactor, LogFunctionFactor,\
                                 ScipyDistributionFactor


_casino_flat_params    = [0.5,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,0.5,0.5]
_gaussian_flat_params  = [0.5,0,1,0,1,0.5,0.5]
_gaussian_flat_params2 = [0.5,-1,1,1,1,0.5,0.5]

# sample HMMs ------------------------------------------------------------------

def get_coins(hmm_type=FirstOrderHMM):
    """Constrruct a two-state HMM with fair and unfair coins

    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    # 0 is fair coin
    # 1 is unfair coin
    #
    # for emissions, 0 is heads, 1 is tails
    state_priors = ArrayFactor(numpy.array([1.0,0.0]))
    emission_factors = [ArrayFactor([0.5,0.5]),
                        ArrayFactor([0.2,0.8])
                        ]
    trans_probs = MatrixFactor(numpy.array([[0.8,0.2],
                                            [0.2,0.8]]))
    return hmm_type(state_priors,emission_factors,trans_probs)

def get_casino(hmm_type=FirstOrderHMM):
    """Construct a two-state HMM over discrete values for testing purposes.
    
    This also implicitly tests ArrayFactor and MatrixFactor

    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    state_priors = ArrayFactor(numpy.array([0.99,0.01]))
    fair   = ArrayFactor(1.0 * numpy.ones(6) / 6.0)
    unfair = ArrayFactor(numpy.array([0.1,0.1,0.1,0.1,0.1,0.5]))
    #unfair = ArrayFactor(numpy.array([0.01,0.01,0.01,0.01,0.01,0.95]))
    emission_probs = [fair,unfair]
    transition_probs = MatrixFactor(numpy.array([[0.95,0.05],
                                                 [0.1,0.9]]))
    casino_hmm = hmm_type(state_priors,
                          emission_probs,
                          transition_probs)
    return casino_hmm


def get_gaussian(hmm_type=FirstOrderHMM):
    """Construct a two-state HMM over continuous distributions for testing purposes

    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    import scipy.stats
    transitions = numpy.matrix([[0.9,0.1],
                                [0.25,0.75]])
    trans_probs = MatrixFactor(transitions)
    state_priors = ArrayFactor([0.8,0.2])
    
    emission_probs = [ScipyDistributionFactor(scipy.stats.norm,loc=0,scale=0.5),
                      ScipyDistributionFactor(scipy.stats.norm,loc=5,scale=10)
                      ]
    return hmm_type(state_priors,emission_probs,trans_probs)

def get_fourstate(hmm_type=FirstOrderHMM):
    """Construct a four-state HMM over discrete values for testing purposes
    
    This also implicitly tests ArrayFactor, MatrixFactor, and ScipyDistributionFactor

    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
    """
    import scipy.stats
    state_priors = ArrayFactor([0.25,0.25,0.25,0.25])
    trans_probs = MatrixFactor(numpy.matrix([[0.8,0.05,0.05,0.1],
                                            [0.2,0.6,0.1,0.1],
                                            [0.01,0.97,0.01,0.01],
                                            [0.45,0.01,0.04,0.5],
                                            ]))
    emission_probs = [ScipyDistributionFactor(scipy.stats.poisson,1),
                      ScipyDistributionFactor(scipy.stats.poisson,5),
                      ScipyDistributionFactor(scipy.stats.poisson,10),
                      ScipyDistributionFactor(scipy.stats.poisson,25),
                      ]
    return hmm_type(state_priors,emission_probs,trans_probs)


# testing functions ------------------------------------------------------------

def test_viterbi(hmm,length=3000,samples=5,do_plot=True,title="",fn=None):
    """Test accuracy of Viterbi decoding algorithm on sequences generated from
    the HMM
    
    @param hmm      FirstOrderHMM instance to test
    
    @param length   Length of sequences to generate
    
    @param fn       File name to save to
    """
    accuracies = numpy.zeros(samples)
    for i in range(samples):
        states, emissions = hmm.generate(length)
        vstates, logprob  = hmm.viterbi(emissions)
        accuracies[i] = float((numpy.array(vstates) == numpy.array(states) ).sum())/len(states)
    
    print "Accuracies (sequence length: %s): %s" % (length,", ".join(["%.2f" % X for X in accuracies]))

    if do_plot is True:
        import matplotlib.pyplot as plt
        from multiplotting import FigurePage
        page = FigurePage()
        ax1, ax2 = page.add_axes_from_grid(columns=1,rows=2,page_margin_bottom=5*72.0)
        page.add_header("%s sample test: accuracy %.2f" % (title,accuracies[-1]))
        ax2.plot(emissions,label="Emissions[:,0]")
        ax1.plot(states,label="True states")
        ax1.plot(0.5+numpy.array(vstates),label="Viterbi states")
        ax2.set_xlabel("Position")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("States")
        ax2.set_ylabel("Arbitrary observable")
        ax1.legend(loc="upper right")
        ax1.set_title("Viterbi decoding")
        ax2.set_title("Emissions")
        ax1.set_xlim(0,200)
        ax2.set_xlim(0,200)
        if fn is not None:
            plt.savefig(fn)
        plt.show()

def test_posterior_decode(hmm):
    pass




# testing functions, applied to example ----------------------------------------

def test_forward_via_coin_eval(length=8):
    """ Compares output of forward algorithm implementations to a manual
    reimplementation that examines all paths in a coin HMM exmaple. Also requires
    that probabilities of all possible observation sequences of a given length
    sum to 1.0
    """
    import itertools
    print "-- Manual value check on algorithms using 10 short sequences in coin example ------------"
    coins = get_coins()
    for c in range(10):
        # generate random sequence of observations
        obs = numpy.random.randint(0,high=2,size=length)
        # evaluate with stock implementation
        assert (coins.fast_forward(obs) - coins.forward(obs)[0]) < 1e-15
        prob = numpy.exp(coins.fast_forward(obs))
    
        # also evaluate brute force by examining every possible path
        manual_prob = 0
        state_paths = itertools.product([0,1],repeat=length)
        for path in state_paths:
            my_prob = coins.state_priors.probability(path[0])*coins.emission_probs[path[0]].probability(obs[0]) 
            for i in range(1,len(path)):
                my_prob *= coins.trans_probs.probability(path[i-1],path[i]) * coins.emission_probs[path[i]].probability(obs[i])
            manual_prob += my_prob

        diff = abs(prob - manual_prob)        
        try:
            assert diff < 1e-15
            assert prob > 0
            print c, prob
        except AssertionError:
            print "    failed test %s: impl %s vs manual %s (diff %s)" % (c,prob, manual_prob, diff)

    # evaluate probability of every sequence of observations
    # should sum to 1.0
    obs_paths = itertools.product([0,1],repeat=length)
    total_prob = 0
    for path in obs_paths:
        total_prob += numpy.exp(coins.fast_forward(path))
    
    print abs(total_prob - 1.0)

    

def test_viterbi_examples(length=3000,samples=5,do_plot=True,namestub="/dev/null/test"):
    print "Testing coins"
    test_viterbi(get_coins(),length=length,samples=samples,do_plot=do_plot,
                 title="Coins",fn="%s_coins.svg" % namestub)
    print "Testing casino"
    test_viterbi(get_casino(),length=length,samples=samples,do_plot=do_plot,
                 title="Casino",fn="%s_casino.svg" % namestub)
    print "Testing gaussian"
    test_viterbi(get_gaussian(),length=length,samples=samples,do_plot=do_plot,
                 title="Gaussian",fn="%s_gaussian.svg" % namestub)
    print "Testing 4-state"
    test_viterbi(get_fourstate(),length=length,samples=samples,do_plot=do_plot,
                 title="Four state",fn="%s_fourstate.svg" % namestub)


def _test_baum_welch_helper(model,
                            samples=100,
                            length=200,
                            max_mse=0.01,
                            initial_params=[],
                            true_params=[],
                            **train_kwargs):
    """Test Baum-Welch on an hmm
    
    Generates observation sequences from the dishonest casino, and then trains
    a new casino from naive or user-supplied parameters
    
    @param model             Generative HMM model
    
    @param samples           Number of observation sequences to generate
    
    @parma length            Length of each observation sequence
    
    @param initial_params    Sequence of initial parameters for untrained model
                              (Default: flat priors)
                              
    @param max_mse           Maximum tolerable mean squared errors between
                             discovered and true parameters  (default: 0.01)
    
    @param true_params       List of true parameters, permuted across equivalent
                             states
                              
    @param train_kwargs      Keyword arguments to pass to training function
    
    @return FirstOrderHMM    Trained hmm
    
    @raises AssertionError   If sum of squared errors between trained and true
                              parameters is greater than the error threshhold
    """
    def get_min_mse(model):
        """Calculate the mean squared error between a model and known true parameters
    
        @return   list<float>  Sum-squared error as compared to parameter set that
                               training most closely approximated
        """
        vals = model.serialize()
        mses = [((vals-numpy.array(X))**2).sum()/len(X) for X in true_params]
        return min(mses)
    
    initial_params = "\t".join([str(X) for X in initial_params])
    printer = sys.stderr
    obs = [model.generate(length)[1] for X in range(samples)]
    naive_hmm = model.from_parameters(initial_params)
    trained_hmm, reason = train_baum_welch(naive_hmm,obs,printer=printer,**train_kwargs)
    min_mse = get_min_mse(trained_hmm)
    print "Training finished due to reason: %s"  % reason
    print "Min mse: %s" % min_mse
    print "Likelihood: %s" % trained_hmm.get_model_log_likelihood(obs)
    assert min_mse < max_mse
    
    return trained_hmm 
    
def _test_baum_welch_casino_helper(model=get_casino(),
                            samples=100,
                            length=200,
                            max_mse=0.01,
                            initial_params=_casino_flat_params,
                            min_iter=50,
                            **train_kwargs):
    """Test Baum-Welch on dishonest casino example
    
    Generates observation sequences from the dishonest casino, and then trains
    a new casino from naive or user-supplied parameters
    
    @param model             Generative HMM model (Default: dishonest casino)
    
    @param samples           Number of observation sequences to generate
    
    @parma length            Length of each observation sequence
    
    @param initial_params    Sequence of initial parameters for untrained model
                              (Default: flat priors)
                              
    @param max_mse           Maximum tolerable mean squared errors between
                             discovered and true parameters  (default: 0.01)
    
    @param train_kwargs      Keyword arguments to pass to training function
    
    @return FirstOrderHMM    Trained hmm
    
    @raises AssertionError   If sum of squared errors between trained and true
                              parameters is greater than the error threshhold
    """
    train_kwargs["emission_estimator"] = DiscreteEmissionEstimator(6)
    train_kwargs["noise_weights"]      = neg_exp_noise_gen(0.1,1)
    true1 = list(model.state_priors.data[:-1]) + list(model.emission_probs[0].data[:-1]) + list(model.emission_probs[1].data[:-1]) + [model.trans_probs.data[0,0],model.trans_probs.data[1,0]]
    true2 = numpy.array([1-true1[0]] + true1[6:11] + true1[1:6] + [1-true1[-1]] + [1-true1[-2]])
    true1 = numpy.array(true1)
    return _test_baum_welch_helper(model,
                                   samples=samples,
                                   length=length,
                                   max_mse=max_mse,
                                   initial_params=initial_params,
                                   true_params=[true1,true2],
                                   min_iter=min_iter,
                                   **train_kwargs)


def _test_baum_welch_gaussian_helper(model=get_gaussian(),
                                     samples=100,
                                     length=200,
                                     max_mse=0.01,
                                     initial_params=_gaussian_flat_params2,
                                     **train_kwargs
                                     ):
    """Test Baum-Welch on dishonest gaussian emissions example
    
    Generates observation sequences from the gaussian HMM, and then trains
    a new hmm from naive or user-supplied parameters
    
    @param samples           Number of observation sequences to generate
    
    @parma length            Length of each observation sequence
    
    @param initial_params    Initial parameters for untrained casino
                              (Default: flat priors)
                              
    @param max_sse           Maximum tolerable sum of squared errors between
                              discovered and true parameters  (default: 0.01)
    
    @param hmm_type          Type of HMM to instantiate (must be FirstOrderHMM
                              or a subclass)
                              
    @param kwargs            Keyword arguments to pass to training function
    
    @return FirstOrderHMM    Trained casino
    
    @raises AssertionError   If sum of squared errors between trained and true
                              parameters is greater than the error threshhold
    """
    #prior 1, mu1, sigma1, mu2, sigma2, trans12, trans22
    #[0.80000000000000004, 0, 0.5, 5, 10, 0.90000000000000002, 0.25]
    true1 = model.serialize()
    true2 = [1-true1[0]] + true1[3:5] + true1[1:3] + [1-true1[-1]] + [1-true1[-2]]
    true1 = numpy.array(true1)
    true2 = numpy.array(true2)
    return _test_baum_welch_helper(model,
                                   samples=samples,
                                   length=length,
                                   max_mse=max_mse,
                                   initial_params=initial_params,
                                   true_params=[true1,true2],
                                   emission_estimator=UnivariateGaussianEmissionEstimator(),
                                   **train_kwargs)


# we can define additional unit tests (e.g. with and without noise) using
# functools.partial and the helpers above
test_baum_welch_via_casino = functools.partial(_test_baum_welch_casino_helper)
test_baum_welch_via_gaussian = functools.partial(_test_baum_welch_gaussian_helper,processes=1)
test_baum_welch_via_casino.__doc__ = _test_baum_welch_casino_helper.__doc__
test_baum_welch_via_gaussian.__doc__ = _test_baum_welch_gaussian_helper.__doc__

