#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:07:04 2020.

@author: joaovitor
"""

import numpy as np


class Audio(object):
    """Audio data interface."""

    def __init__(self, data: np.ndarray, samplerate: int) -> None:
        """
        Audio data objects is a representation of a waveform and a sample rate.

        They hold the basic information to play sound, or record sound to it.

        Parameters
        ----------
        data : np.ndarray
            An array containing audio data, or a buffer to be filled with audio.
        samplerate : int
            Audio sample rate, or how many data represent one second of data.

        Returns
        -------
        None

        """
        self._data = data.reshape((-1, 1)) if data.ndim == 1 else data
        self._samplerate = int(samplerate)
        return

    def __getitem__(self, key):
        """Route AudioData getitem to numpy.ndarray getitem."""
        return np.ndarray.__getitem__(self.data, key)

    def __setitem__(self, key, val):
        """Route AudioData setitem to numpy.ndarray setitem."""
        return np.ndarray.__setitem__(self.data, key, val)

    @property
    def data(self):
        """Audio data as a numpy.ndarray."""
        return self._data

    @property
    def samplerate(self) -> int:
        """Sample rate of the audio."""
        return self._samplerate

    @property
    def nsamples(self) -> int:
        """Total number of data."""
        return self.data.shape[0]

    @property
    def nchannels(self) -> int:
        """Total number of channels."""
        return self.data.shape[1]

    @property
    def samplesize(self) -> int:
        """Size of one sample of audio."""
        return self.data.itemsize

    @property
    def dtype(self) -> np.dtype:
        """Type of the data."""
        return self.data.dtype
