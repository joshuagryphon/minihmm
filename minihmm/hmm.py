#!/usr/bin/env python
"""Contains classes that represent first-order Hidden Markov Models. These
encapsulate methods for:

    #. Assigning state labels to sequences of observations, decoding via
       the Viterbi algorithm or by posterior decoding.

    #. Computing the probability of observing a sequence of observations via
       the forward algorithm

    #. Generating sequences of observations

All classes here support multivariate and univariate emissions (observation
sequences) that can be continuous or discrete.

Training utilities for estimating model parameters may be found in
:mod:`minihmm.training`. Utilities for manipulating higher-order models may be
found in :mod:`minihmm.represent`



References
----------
[Durbin1998]
    Durbin R et al. (1998). Biological sequence analysis: Probabilistic models
    of proteins and nucleic acids. Cambridge University Press, New York.
    ISBN 978-0-521-62971-3

[Rabiner1989]
    Rabiner, LR (1989). A Tutorial on Hidden Markov Models and Selected
    Applications in Speech Recognition. Proceedings of the IEEE, 77(2), pp
    257-286

[WikipediaForwardBackward]
    http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

[WikipediaViterbi]
    http://en.wikipedia.org/wiki/Viterbi_algorithm
"""
import warnings
import copy
import numpy
import jsonpickle
import jsonpickle.ext.numpy
jsonpickle.ext.numpy.register_handlers()

from minihmm.factors import (
    AbstractFactor,
    ArrayFactor,
    MatrixFactor,
    FunctionFactor,
    LogFunctionFactor,
)


class FirstOrderHMM(AbstractFactor):
    """First-order homogeneous hidden Markov model.

    Observations/emissions can be multivariate
    """

    def __init__(self, state_priors=None, emission_probs=None, trans_probs=None):
        """Create a First order hidden Markov model

        Parameters
        ----------
        state_priors : :class:`~minihmm.factors.ArrayFactor`
            Probabilities of starting in any state

        emission_probs  : list of Factors
            Probability distributions describing the probabilities of observing
            any emission in each state. If a list, the types of factors need
            not be identical (e.g. some could be Gaussian, others T-distributed,
            et c)

        trans_probs : :class:`~minihmm.factors.MatrixFactor`
            Matrix describing transition probabilities from each state (first
            index) to each other state (second index).
        """
        splen = len(state_priors)
        tlen = len(trans_probs)
        elen = len(emission_probs)
        if splen != tlen or splen != elen:
            raise ValueError(
                "State priors, transition probabilities, and emission factors "
                "must have same length. Found %s, %s, %s instead."
                % (splen, tlen, elen)
            )

        self.num_states = len(state_priors)
        self.state_priors = state_priors
        self.emission_probs = emission_probs
        self.trans_probs = trans_probs
        self.__logt = None

    @property
    def _logt(self):
        """Log of transition probability matrix"""
        # lazily evaluate log of transition probabilities on first use
        if self.__logt is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "divide by zero encountered in log", RuntimeWarning
                )
                self.__logt = numpy.log(self.trans_probs.data)

        return self.__logt

    @property
    def trans_probs(self):
        return self._trans_probs

    @trans_probs.setter
    def trans_probs(self, value):
        self._trans_probs = value
        # maintain synchrony with log form used in many calculations
        self.__logt = None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<%s, %s states>" % (self.__class__.__name__, self.num_states)

    def get_header(self):
        """Return a list of parameter names corresponding to elements returned
        by :meth:`FirstOrderHMM.get_row`

        Returns
        -------
        list
            List of parameter names
        """
        ltmp = ["sp_%s" % X for X in self.state_priors.get_header()]
        ltmp += ["t_%s" % X for X in self.trans_probs.get_header()]
        for n, e_prob in enumerate(self.emission_probs):
            ltmp += ["e%d_%s" % (n, X) for X in e_prob.get_header()]

        return ltmp

    def get_row(self):
        """Serialize parameters as a list, to be used e.g. as a row in a
        :class:`pandas.DataFrame`

        Returns
        -------
        list
            List of parameter values
        """
        ltmp = self.state_priors.get_row()
        ltmp += self.trans_probs.get_row()
        for e_prob in self.emission_probs:
            ltmp += e_prob.get_row()

        return ltmp

    def to_json(self):
        """Return a string JSON blob encoding `self`"""
        using_funcfactor = False
        for factor in self.emission_probs:
            if isinstance(factor, (FunctionFactor, LogFunctionFactor)):
                warnings.warn(
                    "(Log)FudenctionFactors may only be successfully revived "
                    "from JSON if defined in a model, or if revived before "
                    "scope is lost",
                    UserWarning
                )

        return jsonpickle.encode(self)

    @staticmethod
    def from_json(stmp):
        """Revive a model from a JSON blob

        Parameters
        ----------
        stmp : str
            JSON blob encoding an HMM

        Returns
        -------
        :class:`FirstOrderHMM`
        """
        return jsonpickle.decode(stmp)

    def probability(self, emission):
        """Compute the probability of observing a sequence of emissions. This
        number is likely to undeflow for long sequences.

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        Returns
        -------
        float
            Probability of sequence of emissions
        """
        return numpy.exp(self.logprob(emission))

    def logprob(self, emission):
        """Compute the log probability of observing a sequence of emissions.

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        Returns
        -------
        float
            log probability of sequence of emissions
        """
        return self.fast_forward(emission)

    def fast_forward(self, emissions):
        """Compute the log probability of observing a sequence of emissions.

        More memory efficient implementation of the forward algorithm,
        retaining only the probability of the terminal and penultimate states
        at each step.  This implementation is not useful for posterior
        decoding, which requires the probability of all intermediate states.
        For that purpose, an alternate implementation is provided by
        :meth:`FirstOrderHMM.forward`

        Numerical underflows are prevented by scaling probabilities at each
        step, following the procedure given in Rabiner (1989), "A Tutorial on
        Hidden Markov Models and Selected Applications in Speech Recognition"

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        Returns
        -------
        float
            log probability of sequence of emissions
        """
        log_probability = 0
        T = self.trans_probs.data
        O0 = numpy.diag(
            [self.emission_probs[X].probability(emissions[0]) for X in range(self.num_states)]
        )
        prev_states_scaled = self.state_priors.data.dot(O0)

        for t in range(1, len(emissions)):
            Ot = numpy.diag(
                [self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)]
            )
            f = prev_states_scaled.dot(T.dot(Ot))
            c = f.sum()
            prev_states_scaled = f / c
            log_probability += numpy.log(c)

        return log_probability

    def forward(self, emissions):
        """Calculates the log-probability of observing a sequence of emissions,
        regardless of the state sequence, using the Forward Algorithm. This
        implementation also retains intermediate state information useful in
        posterior decoding or Baum-Welch training.

        Numerical underflows are prevented by scaling probabilities at each step,
        following the procedure given in Rabiner (1989), "A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition"

        Vectorized implementation from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        Returns
        -------
        float
            log probability of sequence of emissions

        numpy.ndarray
            `[time x num_states]` Array representing  scaled forward algorithm
            vector, indicating the forward probability of being in each state
            at time `T`, given the model and the observation trajectory from
            `t = 0` to `t = T`

        numpy.ndarray
            `[time x 1]` Array of scaling constants used at each step as
            described in Rabiner 1989.  The sum of the log of these equals the
            log probability of the observation sequence.
        """
        total_logprob, scaled_forward, _, scale_factors, _ = self.forward_backward(emissions, calc_backward=False)
        return total_logprob, scaled_forward, scale_factors

    def forward_backward(self, emissions, calc_backward=True):
        """Calculates the forward algorithm, the backward algorithm, and
        sufficient statistics useful in Baum-Welch calculations, all in
        factored and vectorized forms.

        Numerical underflows are prevented by scaling forward  probabilities at
        each step, following the procedure given in Rabiner (1989), "A Tutorial
        on Hidden Markov Models and Selected Applications in Speech
        Recognition"

        Vectorized implementation adapted from Wikipedia (2014-02-20):
        http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm but
        using scaling as described in Rabiner1989 and Durbin1998, rather than
        the Wikipedia article


        Parameters
        ----------
        emissions : numpy.ndarray
            Observations

        calc_backward : bool, optional
            If True, perform backward pass and calculate Rabiner's ksi table


        Returns
        -------
        float
            log probability of sequence of emissions

        numpy.array
            Scaled forward algorithm vector of dim `[time x num_states]`,
            indicating the forward probability of being in each state at time
            `T`, given the model and the observation trajectory from `t = 0` to
            `t = T`

        numpy.ndarray
            Scaled backward algorithm vector of dim `[time x num_states]`,
            indicating the backward probability of being in each state at time
            `T`, given the model and the observation trajectory from `t = T` to
            `t = end`

        numpy.ndarray
            `[time x 1]` Array of scaling constants used at each step as
            described in Rabiner 1989.  The sum of the log of these equals the
            log probability of the observation sequence.

        numpy.ndarray
            `[time x num_states x num_states]` ksi table, as described in
            Rabiner 1989.  At each time `t`, `ksi[t,i,j]` gives the posterior
            probability of transitioning from state `i` to state `j`. From this
            table it is trivial to derive the expected number of transitions
            from state `i` to state `j`, or the posterior probability of being
            in state `i` or `j` at a given timepoint, by taking the appropriate
            sum.


        Notes
        -----
        This implementation casts everything to ``numpy.float128``. Whether
        this will actually force use of IEEE float128 depends on local C
        library implementations
        """
        # probability sequence indexed by timeslice. columns are end states
        scaled_forward = numpy.tile(numpy.nan, (len(emissions), self.num_states))
        scale_factors = numpy.ones(len(emissions))
        T = self.trans_probs.data
        O = []

        # initialize as prior + likelihood of emissions
        O.append([self.emission_probs[X].probability(emissions[0]) for X in range(self.num_states)])
        scaled_forward[0, :] = self.state_priors.data.dot(numpy.diag(O[0]))

        # can get underflows here from very improbable emissions
        #
        # can get nan for probability if f.sum() is 0, in other words, if a
        # given observation is very improbable for all models and underflows
        # for all models, then c = 0, and f/c = [nan,nan,...,nan]
        #
        # This then forces all future probabilities to be set to nan,
        # which messes up forward and backward calculations.
        # In this case, using HighPrecisionFirstOrderHMM will work,
        # but at a cost for speed
        for t in range(1, len(emissions)):
            O.append(
                [self.emission_probs[X].probability(emissions[t]) for X in range(self.num_states)]
            )
            f = scaled_forward[t - 1, :].dot(T.dot(numpy.diag(O[t])))
            c = f.sum()
            scaled_forward[t, :] = f / c
            scale_factors[t] = c

        if calc_backward is True:
            # backward calc
            scaled_backward = numpy.zeros((len(emissions), self.num_states))
            scaled_backward[-1, :] = 1.0 / scale_factors[
                -1
            ]  # <---- Wikipedia says not to scale final timestep; Rabiner & Durbin say to
            for t in range(len(emissions) - 1)[::-1]:
                scaled_backward[t, :] = T.dot(numpy.diag(O[t + 1]).dot(scaled_backward[t + 1, :])
                                              ) / scale_factors[t]

            # ksi calc
            # NOTE: this is a complete calculation despite the fact that we
            #       are working in a scaled space, because the scale factors
            #       end up equaling 1/P(O|lambda), which means we can compute
            #       in scaled space and get an unscaled result if we do not
            #       divide by P(O|lambda)
            O = numpy.array(O)
            ksi = scaled_forward[:-1, :, None] * scaled_backward[1:, None, :] * T[None, :, :
                                                                                  ] * O[1:, None, :]

        else:
            scaled_backward = None
            ksi = None

        if numpy.isnan(scale_factors).any():
            total_logprob = -numpy.Inf
        else:
            total_logprob = numpy.log(scale_factors).sum()

        return total_logprob, scaled_forward, scaled_backward, scale_factors, ksi

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
            An array of dimension `[t x 1]` of the most likely states at each
            point `t`

        numpy.ndarray
            An array of dimension `[t x k]` of the posterior probability of
            being in state `k` at time `t`
        """
        _, forward, backward, scale_factors, _ = self.forward_backward(emissions)
        posterior_probs = forward * backward * scale_factors[:, None]
        most_likely_states = (posterior_probs).argmax(1)

        return most_likely_states, posterior_probs

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
        generative. I.e. if using :class:`~minihmm.factors.FunctionFactor` or
        :class:`~minihmm.factors.LogFunctionFactor`, generator functions must
        be specified at their instantiation.
        """
        states = []
        emissions = []

        states.append(self.state_priors.generate())
        emissions.append(self.emission_probs[states[0]].generate())

        logprob = self.state_priors.logprob(states[0])
        logprob += self.emission_probs[states[0]].logprob(emissions[0])

        for i in range(1, length):
            new_state = self.trans_probs.generate(states[i - 1])
            new_obs = self.emission_probs[new_state].generate()

            states.append(new_state)
            emissions.append(new_obs)

            logprob += self.trans_probs.logprob(states[-1], new_state)
            logprob += self.emission_probs[new_state].logprob(new_obs)

        return numpy.array(states).astype(int), numpy.array(emissions), logprob

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
        T = self._logt
        joint_logprob = self.state_priors.logprob(path[0])
        joint_logprob += self.emission_probs[path[0]].logprob(emissions[0])

        for i in range(1, len(emissions)):
            joint_logprob += T[path[i - 1], path[i]] + self.emission_probs[path[i]].logprob(
                emissions[i]
            )

        return joint_logprob

    def conditional_path_logprob(self, path, emissions):
        """Return log P(path | emissions) evaluated under this model

        Parameters
        ----------
        path : list-like
            Sequence of states

        emissions : list-like
            Sequence of observations

        Returns
        -------
        float
            Log probability of `P(path | emissions)`
        """
        # calculate conditional pat logprob as P(path, emissions) - P(emissions)
        # P(emissions) from fast_forward
        obs_logprob = self.fast_forward(emissions)
        joint_logprob = self.joint_path_logprob(path, emissions)

        return joint_logprob - obs_logprob

    def sample(self, emissions, num_samples=1):
        """Sample state sequences from the distribution `P(states | emissions)`,
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
        L = len(emissions)
        T = self.trans_probs.data

        total_logprob, scaled_forward, _, scale_factors, _ = self.forward_backward(
            emissions, calc_backward=False
        )
        randos = numpy.random.random(size=(num_samples, L))
        final_state_cumsums = scaled_forward[-1, :].cumsum()

        paths = []
        for n in range(num_samples):
            my_path = numpy.full(len(emissions), -1, dtype=int)

            # because probabilty at all steps is scaled to one, we can just
            # examine cumsum of final step to start
            last_state = final_state_cumsums.searchsorted(randos[n, 0], side="right")
            #last_state = (scaled_forward[-1,:].cumsum() >= randos[n,0]).argmax()
            my_path[-1] = last_state

            for i in range(1, L):
                pvec = T[:, last_state] \
                       * self.emission_probs[last_state].probability(emissions[-i]) \
                       * scaled_forward[-i-1, :] \
                       / scaled_forward[-i, last_state] \
                       / scale_factors[-i]

                #last_state = (pvec.cumsum() >= randos[n, i]).argmax()
                last_state = pvec.cumsum().searchsorted(randos[n, i], side="right")
                my_path[-i - 1] = last_state

            paths.append(my_path)

        return paths


#     commented out - this actually samples according to joint distribution P(states,obs), NOT P(states|obs)
#     def sample(self, emissions):
#         """Probabilistically sample state sequences from the joint distribution
#         P(emissions, states), using a modified Viterbi algorithm
#
#         See Durbin1997 ch 4.3
#
#         Parameters
#         ----------
#         emissions : numpy.ndarray
#             Sequence of observations
#
#         Returns
#         -------
#         dict
#             Results of decoding, with the following keys:
#
#             `states`
#                 :class:`numpy.ndarray`. Decoded labels for each position in
#                 emissions[start:end]
#
#             `logprob`
#                 :class:`float`. Joint log probability of sequence of
#                 `emissions[start:end]` and the decoded state sequence
#         """
#         rvals = numpy.random.random(len(emissions))
#         T     = self._logt
#
#         prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[0])\
#                                   for X in range(self.num_states)])
#
#         newstate   = (prev_probs.cumsum() >= rvals[0]).argmax()
#         state_path = [newstate]
#         logprob    = prev_probs[newstate]
#
#         for r, x in zip(rvals, emissions)[1:]:
#             emission_logprobs = numpy.array([X.logprob(x) for X in self.emission_probs])
#             current_probs     = T[state_path[-1],:] + emission_logprobs + logprob
#             newstate          = (current_probs.cumsum() >= r).argmax()
#
#             state_path.append(newstate)
#             logprob += current_probs[newstate]
#
#         dtmp = {
#             "sampled_states"  : state_path,
#             "logprob"         : logprob,
#         }
#         return dtmp

    def viterbi(self, emissions, start=0, end=None):
        """Finds the most likely state sequence underlying a set of emissions
        using the Viterbi algorithm. Also returns the natural log probability
        of that state sequence.

        See http://en.wikipedia.org/wiki/Viterbi_algorithm

        Parameters
        ----------
        emissions : numpy.ndarray
            Sequence of observations

        start : int, optional
            Starting point in emissions to consider (Default: 0)

        end : int, optional
            Ending point in emissions to consider (Default: None, end of sequence)


        Returns
        -------
        dict
            Results of decoding, with the following keys:

            `viterbi_states`
                :class:`numpy.ndarray`. Decoded labels for each position in
                emissions[start:end]

            `logprob`
                :class:`float`. Joint log probability of sequence of
                `emissions[start:end]` and the decoded state sequence

            `state_paths`
                dict[state] = log probability of final symbol being in that state
        """
        state_dict = {K: [K] for K in range(self.num_states)}
        prev_probs = numpy.array([self.state_priors.logprob(X) + self.emission_probs[X].logprob(emissions[start]) \
                                  for X in range(self.num_states)])

        T = self._logt
        for x in emissions[start + 1:end]:
            new_state_dict = {}
            emission_logprobs = numpy.array([X.logprob(x) for X in self.emission_probs])
            current_probs = T + emission_logprobs[None, :] + prev_probs[:, None]

            new_probs = current_probs.max(0)
            new_state_dict = { X : state_dict[current_probs[:, X].argmax()] + [X] \
                                   for X in range(self.num_states) }

            prev_probs = new_probs
            state_dict = new_state_dict

        final_label = prev_probs.argmax()
        total_logprob = prev_probs.max()
        states = numpy.array(state_dict[final_label])

        dtmp = {
            "viterbi_states": states,
            "state_paths": state_dict,
            "logprob": total_logprob,
        }
        return dtmp
