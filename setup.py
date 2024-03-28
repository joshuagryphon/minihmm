#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.rst") as f:
    long_description = f.read().replace(":mod:", "")


config_info = {
    "version"  : "0.3.3",
    "packages" : find_packages(),
}


setup(
    name = "minihmm",
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "jsonpickle",
    ],

    # metadata for upload to PyPI
    author           = "Joshua Griffin Dunn",
    maintainer       = "Joshua Griffin Dunn",
    
    description = (
        "Lightweight extensible HMM engine, supporting univariate or "
        "multivariate, continuous or discrete emissions"
    ),
    long_description = long_description,
    long_description_content_type = "text/x-rst",

    license   = "BSD 3-Clause",
    keywords  = "HMM hidden Markov model machine learning modeling statistics",
    url       = "http://github.com/joshuagryphon/minihmm",
    platforms = "POSIX",
    
    classifiers=[
         'Development Status :: 4 - Beta',
         'Programming Language :: Python',
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.9',
         'Topic :: Scientific/Engineering',
         'License :: OSI Approved :: BSD License',
         'Operating System :: POSIX',
        ],
    
    **config_info
) # yapf: disable
