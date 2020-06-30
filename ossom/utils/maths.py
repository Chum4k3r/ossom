# -*- coding: utf-8 -*-
"""
Provide mathematical functions set up to process data from Audio objects.

Created on Tue May  5 00:34:36 2020

@author: João Vitor Gutkoski Paes
"""

import numpy as np
import numba as nb


@nb.njit(parallel=True)
def apply_channelwise(auddata: np.ndarray, func: callable) -> np.ndarray:
    """
    Apply func on each column of an audio data array.

    Parameters
    ----------
    auddata : np.ndarray
        Audio data as numpy array.
    func : callable
        The function to apply on each channel. Must receive data as a 1d array.

    Returns
    -------
    chwise : np.ndarray
        Output of func for each channel as an array.

    """
    chwise = np.zeros((1, auddata.shape[1]), dtype=auddata.dtype)
    for channel in nb.prange(chwise.shape[1]):
        chwise[:, channel] = func(auddata[:, channel])
    return chwise



@nb.njit
def max_abs(auddata: np.ndarray) -> np.ndarray:
    """
    Maximum of the absolute values of the input array, for each channel (column).

    See `apply_channelwise` for more info

    Parameters
    ----------
    auddata : np.ndarray
        Input array of audio data.

    Returns
    -------
    ma : np.ndarray
        Columnwise `max` of `abs` of `arr`.

    """
    ma = apply_channelwise(auddata, lambda arr: np.max(np.abs(arr)))
    return ma


@nb.njit
def rms(auddata: np.ndarray) -> np.ndarray:
    """
    Root of the mean of the squared values of input array, for each channel.

    See `apply_channelwise` for more info.

    Parameters
    ----------
    auddata : np.ndarray
        Input array of audio data.

    Returns
    -------
    rms : np.ndarray
        Columnwise `sqrt` of `mean` of `arr**2`.

    """
    rms = apply_channelwise(auddata, lambda arr: np.mean(arr**2)**0.5)
    return rms


@nb.njit
def dB(auddata: np.ndarray, power: bool = False, ref: float = 1.0) -> np.ndarray:
    """
    Decibel level of input array, for each channel.

    See `apply_channelwise` and `rms` for more info.

    Parameters
    ----------
    auddata : np.ndarray
        Input array of audio data.
    power : bool, optional
        If True, assumes a power signal, which need not to be squared. The default is False.
    ref : float, optional
        The decibel reference. The default is 1.0.

    Returns
    -------
    _np.ndarray
        Channelwise audio levels, in decibel.

    """
    return (10 if power else 20) * np.log10(auddata / ref)


@nb.njit
def noise(level: float, samplerate: int, tlen: float, nchannels: int) -> np.ndarray:
    """
    Generate random noise for audio data.

    Parameters
    ----------
    level : float
        Level of audio.
    samplerate : int
        Sample rate.
    tlen : float
        Total time length.
    nchannels : int
        Total number of channels.

    Returns
    -------
    _np.ndarray
        The complete noise data.

    """
    shape = (int(samplerate*tlen), nchannels)
    noise = np.zeros(shape, dtype=np.float32)
    noise[:] = np.random.randn(*shape)
    noise[:, ] /= max_abs(noise)
    return level*noise    # TODO: DEVELOP FOR DECIBEL GAIN LEVEL.

