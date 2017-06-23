#!/usr/bin/env python
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
)

from nose.tools import (
    assert_equal,
    assert_true,
    assert_dict_equal,
    assert_list_equal,
    assert_tuple_equal,
    assert_raises,
)

def check_equal(a, b, msg=None):
    assert_equal(a, b, msg)

def check_array_equal(a, b, **kwargs):
    assert_array_equal(a, b, **kwargs)

def check_almost_equal(a,b,kwargs={}):
    assert_almost_equal(a,b, **kwargs)

def check_true(a, kwargs={}):
    assert_true(a, **kwargs)

def check_not_equal(a, b):
    assert_raises(AssertionError, check_array_equal, a, b)

def check_list_equal(a, b, msg=None):
    assert_list_equal(a, b, msg)

def check_dict_equal(a, b, msg=None):
    assert_dict_equal(a, b, msg)

def check_tuple_equal(a, b, msg=None):
    assert_tuple_equal(a, b, msg)

def check_raises(cls, callable_, *args):
    assert_raises(cls, callable_, *args)
