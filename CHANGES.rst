Change Log
==========
Changes to ``minihmm`` are documented here.

Version numbers follow following the conventions described in `PEP440
<https://www.python.org/dev/peps/pep-0440/>`_ and `Semantic versioning
<http://semver.org/>`_. Because we're still below v1, minor updates might
break backward compatibility.


v0.3.0 = [2018-08-28]
---------------------

- BREAKING: footprint of serialized objects now smaller

- Improved screen output in ``DefaultLoggerFactory()``

- Improved code styling


v0.2.1 = [2017-11-26]
---------------------

- Restored Python 3 compatibility


v0.2.0 = [2017-11-25]
---------------------

- Migrated serialization setup to ``jsonpickle``, a practice much saner than
  what I had been previously doing. This breaks backward compatibility with
  v0.1.4 JSON blobs, but enables far more flexibility in what can be serialized

- replace older ``(de)serialize()`` methods with much saner ``get_header()``
  and ``get_row()`` for export of models as ``DataFrame`` rows (e.g. to watch
  parameter trajectories during training

- ``train_baum_welch()`` can now weight individual observations

- changed logging in ``train_baum_welch()``. This led to some
  backward-incompatible changes, but is much nicer. It also includes a
  ``DefaultLoggerFactory()`` so people don't have to comb through insane
  logs anymore

- Unit tests and multiple improvments for ``train_baum_welch()``



v0.1.4 = [2017-06-08]
---------------------

- Some class properties are now manged, saving me from myself

- ``to_dict()`` and ``from_dict()`` methods specified for serializing models as
  JSON blobs

- Convenience methods for building HMM tables from known observations,
  optionally with weights

- Speed improvements under the hood

- Suppression of non-useful warnings, and creation of useful ones

- Unit tests for key features



v0.1.3 = [2017-05-09]
---------------------

- Model reduction tested and working, even though unit tests not yet 
  fleshed out

- Valid pseudocount arrays now generated for state priors in high order space
  (before was only for transition tables)

- Added warnings in places where unexpected side effects could be caused by
  valid calculations (e.g. uncaught `nan`)

- Serialization sketched out for FirstOrderHMMs



v0.1.2 = [2017-05-08]
---------------------

Added
......

- Methods for serializing and deserializing large matrices to JSON

- Methods for reducing high-order models to first-order models, and
  for converting state sequences between orders

- Yet more unit tests



v0.1.1
------

Added
.....

- Can now sample state paths given an observation sequence, from the
   conditional distribution :math:`P(Path | observation sequence)`

- Unit tests


Changed
.......

- miniHMM factored out of unpublished scientific project

- Migration from SVN to GIT repo

- get_free_parameters() and from_parameters() replaced by serialize()
  and deserialize() methods in all factors
