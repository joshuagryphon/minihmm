miniHMM
=======

This is a toy library that implements first-order homogeneous hidden Markov
models.  I have found it useful for a number of pet projects, but I am in the
process of factoring it out of another project and simultaneously rewriting
substantial pieces of it, so it should not be considered remotely stable by
anybody who happens upon it. In fact, it just *might* be broken right now!

At present, the following are implemented:

 - Creation of models with univariate or multivariate, discrete or continuous
   emissions

 - Model training via a Baum-Welch implementation with some bells & whistles
   (e.g.  noise scheduling, parallelization, et c)

 - Decoding / inference of state paths, via the Viterbi algorithm or posterior
   decoding

 - Sampling state paths according to their conditional probability, given a
   sequence of observations

 - Aaaaaaaand some other stuff. I swear.


Things to look forward to in the future:

 - better serialization/storage of models

 - high order models!

 - non-hidden Markov models

Cheers,
Josh
