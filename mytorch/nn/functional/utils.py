"""Helpers to unwrap :class:`Tensor` / :class:`Array` to raw NumPy/CuPy buffers."""
from __future__ import annotations

import numpy as np

from ..._array import Array
from ...tensor import Tensor

try:
    import cupy as cp
    _NDARRAY_TYPES = (np.ndarray, cp.ndarray)
except ImportError:
    cp = None
    _NDARRAY_TYPES = (np.ndarray,)


def get_inner_array(arr):
    """Return the backing ``numpy.ndarray`` or ``cupy.ndarray`` for ``arr``.

    Parameters
    ----------
    arr : Tensor, Array, or ndarray
        Value to unwrap.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
    """
    if isinstance(arr, Tensor):
        arr = arr.data
    if isinstance(arr, Array):
        return arr._array
    if isinstance(arr, _NDARRAY_TYPES):
        return arr
    raise TypeError(f"Expected Tensor, Array, or ndarray, got {type(arr)!r}")


def get_inner_inner_array(arr):
    """Alias for :func:`get_inner_array` (historical name)."""
    return get_inner_array(arr)
