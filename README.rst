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
  continuous or discrete. See :mod:`minihmm.factors` for examples of
  distributions that can be built out-of-the-box, and for hints on designing new
  ones,
  
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


Running the tests
-----------------

Tests are currently written to run under :mod:`nose` separately under Python 3.6
and 3.9, with the following virtual environments configured via :mod:`tox`:

- `*-pinned` : run using versions of dependencies pinned in ``requirements.txt``

- `*-latest` : run all tests using latest available versions of each dependency.
  This will enable us to catch breaking changes.

By default, running ``tox`` from the shell will run all tests in all
environments. To choose which environment(s) or test(s) to run, you can use
standard :mod:`tox` or :mod:`nose` arguments (see their respective documentation
for more details)::

    # run tests only under Python 3.6, with pinned requirements
    $ tox -e py36-pinned 

    # run tests under all environments, but only for estimator suite
    $ tox minihmm.test.test_estimators

    # run tests only for estimator suite, passing verbose mode to nose
    # note: nose args go after the double dash ('--')
    $ tox minihmm.test.test_estimators -- -v --nocapture


As these environments assume you have Python 3.6, and 3.9 installed, we have
defined a Dockerfile that contains all of them. This is the preferred
environment for testing. Build the image with the following syntax::

    # build image from inside miniHMM folder
    $ docker build --pull -t minihmm .

    # start a container, mounting current folder as minihmm source
    $ docker run -it --rm minihmm

    # alternative if you are developing- mount your dev folder within
    # the container, then run tox inside the container
    $ docker run -it --rm $(pwd):/usr/src/minihmm minihmm


Building the documentation
--------------------------

Documents may be built via Sphinx, either inside or outside the container.
To build the docs, you must first install the package, as well as documentation
dependencies. In the project folder::

    # install package
    $ pip install --user -e .

    # install doc dependencies
    $ pip install -r docs/requirements.txt

    # build docs & open in browser
    $ make -C docs html
    $ firefox docs/build/html/index.html


Notes
-----

This library is in beta, and breaking changes are not uncommon. We try to be
polite by announcing these in the changelog.
