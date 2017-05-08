Change Log
==========
Changes to ``minihmm`` are documented here.

Version numbers follow a scheme of ERA.MAJOR.MINOR, following the conventions
described in `PEP440 <https://www.python.org/dev/peps/pep-0440/>`_ and 
`Semantic versioning <http://semver.org/>`_, with the exception that we have
prepended the era to the version number.


Unreleased
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
