#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:40:13 2020

@author: joaovitor
"""


import numpy as _np
import warnings
from typing import List


class Configurations:
    """Module-wide configurations."""

    _samplerate: int = 48000
    _blocksize: int = 512
    _buffersize: int = 480000
    _dtype: _np.dtype = _np.float32().dtype
    _channels: List[int] = [0, 1]

    _instance = None

    def __init__(self):
        return

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @samplerate.setter
    def samplerate(self, new: int):
        if 11025 <= new and new <= 192000:
            new = int(new)
            self._samplerate = new
        else:
            raise ValueError("Sample rate out of acceptable range [11025, 192000].")
        return

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @blocksize.setter
    def blocksize(self, new: int):
        self._blocksize = int(new)
        return

    @property
    def buffersize(self) -> int:
        return self._buffersize

    @buffersize.setter
    def buffersize(self, new: int):
        if new < self.samplerate * 5:
            warnings.warn("Buffer may be too short.", ResourceWarning)
        self._buffersize = int(new)
        return

    @property
    def dtype(self) -> _np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, new: str or _np.dtype):
        new = _np.dtype(new).name
        if new == 'float64':
            new = 'float32'
        self._dtype = _np.dtype(new)
        return

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, new: List[int]):
        if type(new) is not list:
            if type(new) is int:
                new = [new]
            else:
                raise ValueError('Channels must be a list of integer values, e.g. [0, 2, 3]')
        self._channels = list(new)
        return

config = Configurations()
