#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension
from Cython.Build import cythonize


with open("README.rst") as f:
    long_description = f.read()


config_info = { "version"      : "0.1.3",
                "packages"     : find_packages(),
              }

ext_modules = [Extension("minihmm.cytest", [os.path.join("minihmm", "cytest.pyx")], language="c")]

setup(
    name = "minihmm",
    build_requires = [
        "cython",
        "numpy>=1.5",
    ],
    install_requires = [
                        "cython",
    	                "numpy>=1.5",
                        "scipy>=0.12.0",
                        "matplotlib>=1.3.0",
                        ],

    ext_modules = ext_modules,
    cmdclass = {
        "build_ext" : build_ext,
    },
    zip_safe = True,

    # metadata for upload to PyPI
    author           = "Joshua Griffin Dunn",
    author_email     = "joshua.g.dunn@gmail.com",
    maintainer       = "Joshua Griffin Dunn",
    maintainer_email = "Joshua Griffin Dunn",
    
    description = "Lightweight extensible HMM engine, supporting univariate or multivariate, continuous or discrete emissions",
    long_description = long_description,
    license   = "BSD 3-Clause",
    keywords  = "HMM hidden Markov model machine learning modeling statistics",
    url       = "http://github.com/joshuagryphon/minihmm",
    platforms = "POSIX",
    
    tests_require=["nose>=1.0"],
    #test_suite = "nose.collector",
    
    classifiers=[
         'Development Status :: 3 - Alpha',

         'Programming Language :: Python',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3.3',
         'Programming Language :: Python :: 3.4',

         'Topic :: Scientific/Engineering',
         'License :: BSD 3-Clause',
         'Operating System :: POSIX',
        ],
    
    **config_info
)
