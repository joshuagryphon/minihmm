from numpy.testing import assert_array_equal
from nose.tools import (
    assert_equal,
    assert_almost_equal,
    assert_dict_equal,
    assert_list_equal,
    assert_tuple_equal,
    assert_raises,
    )


def check_equal(a, b, msg=None):
    assert_equal(a, b, msg)

def check_list_equal(a, b, msg=None):
    assert_list_equal(a, b, msg)

def check_dict_equal(a, b, msg=None):
    assert_dict_equal(a, b, msg)

def check_array_equal(a, b, msg=None):
    assert_array_equal(a, b, msg)

def check_tuple_equal(a, b, msg=None):
    assert_tuple_equal(a, b, msg)

def check_raises(cls, callable_, *args):
    assert_raises(cls, callable_, *args)

