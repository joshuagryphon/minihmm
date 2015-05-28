#!/usr/bin/env python
"""Utilities used by multiple functions
"""
class NullWriter(object):
    """File-like object that actually writes nothing, in the spirit of :obj:`os.devnull`
    """

    def write(inp):
        pass

    def __repr__(self):
        return "NullWriter()"
    
    def __str__(self):
        return "NullWriter()"
