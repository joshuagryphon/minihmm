miniHMM
=======

This is a toy library that implements first-order homogeneous hidden Markov
models.  I have found it useful for a number of pet projects, but I am in the
process of factoring it out of another project and simultaneously rewriting
substantial pieces of it, so it should not be considered remotely stable by
anybody who happens upon it. In fact, it just *might* be broken right now!

At present, it does offer some benefits hard to find in other libraries:

 - It contains utilities to translate *high-order models* into first-order space,
   enabling modeling of larger local interactions. High-order models, translated
   into first-order space, are mathematically equivalent to native high-order
   models, but moving them into first-order space enables them to be trained and
   used for inference using classic algorithms (see ``minihmm.representation``).

 - It's architecture is modular and flexible, enabling emissions to be
   univariate or multivariate (for multidimensional cases). In addition,
   emission can be continuous, or discrete. See ``minihmm.factors`` for
   examples of distributions that can be built, and ``minihmm.estimators``
   for examples of classes that can be used to re-estimate distributions during
   training.
   
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

 - Aaaaaaaand some other stuff. I swear.



Cheers,
Josh
