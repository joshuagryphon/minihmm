#!/usr/bin/env python
"""Utilities used by multiple functions / across the library
"""
import numpy
from scipy.sparse import coo_matrix

#===============================================================================
# Printing
#===============================================================================


class NullWriter(object):
    """File-like object that actually writes nothing, in the spirit of :obj:`os.devnull`
    """

    def write(inp):
        pass

    def __repr__(self):
        return "NullWriter()"
    
    def __str__(self):
        return "NullWriter()"
    
    def close(self):
        pass

    def flush(self):
        pass


#===============================================================================
# Building transition tables
#===============================================================================

def build_transition_table(num_states,
                           state_sequences,
                           weights          = None,
                           pseudocounts     = 0,
                           normalize        = True):
    """Build a set of transition tables from sequences of known states.  This
    in contrast to *training*, in which parameters for transition tables are
    estimated from msequences of observations and unknown states.
    
    Parameters
    ----------
    num_states : int
        Number of states

    state_sequences : list of list_like
        Sequences of states, represented as integers

    weights : list-like or None, optional
        Weight to apply to each sequence. If `None`, each sequence will be
        weighted equally (Default: `None`)

    pseudocounts : int or matrix-like
        Pseudocounts to add to count table. If `int`, same value will be added to every
        cell in the table. If matrix or array-like, that matrix will be added to the 
        count matrix (Default: `0`)

    normalize : bool, optional
        If `True`, return a row-normalized transition table. If `False`, return a count
        table (Default: `True`)
    """
    tmat = numpy.zeros((num_states, num_states), dtype=int)
    if weights is None:
        weights = [1] * len(state_sequences)

    for my_seq, my_weight in zip(state_sequences, weights):
        for i in range(len(my_seq) - 1):
            from_seq, to_seq = my_seq[i:i+2]
            tmat[from_seq, to_seq] += my_weight

    tmat += pseudocounts
    if normalize == True:
        tmat = (1.0 * tmat.T / tmat.sum(1)).T

    return tmat


#===============================================================================
# Serialization / deserialization of matrices
#===============================================================================

def matrix_to_dict(mat):
    """Convert a matrix or array `mat` to a dictionary.
    This is intende to be useful for serializing large, sparse matrices of
    high-order hidden Markov models.

    Parameters
    ----------
    mat : :class:`numpy.ndarray`, or something like it


    Returns
    -------
    dict
        Dictionary with the following properties:

        `shape`
            A tuple of matrix dimensions

        `row`
            A list of row coordinates

        `col`
            A list of column coordinates

        `data`
            A list of values

    See also
    --------
    matrix_from_dict
    """
    coomat = coo_matrix(mat)
    dout = {
        "shape" : tuple(coomat.shape),
        "row"   : list(coomat.row.astype(int)),
        "col"   : list(coomat.col.astype(int)),
        "data"  : list(coomat.data.astype(float)),
    }
    return dout

def matrix_from_dict(dtmp, dense=False):
    """Reconstruct a matrix from a dictionary made e.g. by :func:`matrix_to_dict`


    Parameters
    ----------
    dict : dict
        Dictionary with keys `shape`, `row`, `col`, and `data`

    dense : bool, optional
        Whether or not to return a dense matrix


    Returns
    -------
    :class:`scipy.sparse.coo_matrix` if `dense` is `False`, otherwise :class:`numpy.Matrix`


    See also
    --------
    matrix_to_dict
    """
    coomat = coo_matrix((dtmp["data"], (dtmp["row"],dtmp["col"])), shape=dtmp["shape"])
    return coomat.todense() if dense is True else coomat



