# -*- coding: utf-8 -*-
"""

Created on Tue May  5 00:34:36 2020

@author: joaovitor
"""

import numpy as _np
import numba as _nb


@_nb.njit(parallel=True)
def max_abs(arr: _np.ndarray) -> _np.ndarray:
    ma = _np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    for col in _nb.prange(ma.shape[1]):
        ma[:, col] = _np.max(_np.abs(arr[:, col]))
    return ma


@_nb.njit(parallel=True)
def rms(samples: _np.ndarray) -> _np.ndarray:
    rms = _np.zeros((1, samples.shape[1]), dtype=samples.dtype)
    for col in _nb.prange(rms.shape[1]):
        rms[:, col] = _np.mean(samples[:, col]**2)**0.5
    return rms

@_nb.njit
def dB(samples: _np.ndarray, power: bool = False, ref: float = 1.0) -> _np.ndarray:
    return (10 if power else 20) * _np.log10(rms(samples) / ref)
