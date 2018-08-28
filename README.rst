miniHMM
=======

Summary
-------

This is a toy library that implements first- through Nth-order hidden Markov
models. 

At present, `miniHMM` offers some benefits hard to find in other HMM libraries:

- Its algorithms are numerically stable

- It is able to compute high order hidden Markov models, which allow states
  to depend on the Nth previous states, rather than only on the immediate
  previous state. 
  
  Concretely, high-order models are implemented via a translation layer
  that converts high-order models of arbitrary degree into mathematically
  equivalent first-order models over a virtual state space. This implementation
  allows all algorithms developed for first-order models to be applied in 
  higher dimensions. See :mod:`minihmm.represent` for further detail.

- Emissions may be univariate or multivariate (for multidimensional emissions),
  continuous or discrete. See ``minihmm.factors`` for examples of distributions
  that can be built out-of-the-box, and for hints on designing new ones,
  
- Multiple distinct estimators are available for probability distributions,
  enabling e.g. addition of model noise, pseudocounts, et c during model
  training. See :mod:`minihmm.estimators` for details.
   
- HMMs of all sorts can be trained via a Baum-Welch implementation with some
  bells & whistles (e.g.  noise scheduling, parallelization, parameter-tying
  (via estimator classes), et c)

- In addition to the Viterbi algorithm (the maximum likelihood solution for a
  total sequence of states), states may be inferred by:
   
  - Probabilistically sampling valid sequences from their posterior
    distribution, given a sequence of emissions. This enables estimates of
    robustness and non-deterministic samples to be drawn

  - Labeling individual states by highest posterior probabilities (even
    though this doesn't guarantee a valid path)


Notes
-----

This library is in beta, and breaking changes are not uncommon. We try to be
polite by announcing these in the changelog.
