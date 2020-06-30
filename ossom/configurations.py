# -*- coding: utf-8 -*-
"""
This module provide a singleton object to hold configurations such as `samplerate`, `blocksize` and any relevant parameter that is necessary to the main `Recorder` and `Player` classes.

These values are used as default case and are available to any OsSom class that might need them.

Created on Sun Jun 28 21:40:13 2020

@author: João Vitor Gutkoski Paes
"""


import numpy as _np
import warnings
from typing import Dict, List


_default = {
    'samplerate': 48000,
    'blocksize': 512,
    'buffersize': 480000,
    'dtype': _np.dtype('float32'),
    'channels': {'in': [0, 1],
                 'out': [0, 1]}
}


class Configurations:
    """Global OsSom configurations."""

    _samplerate: int = None
    _blocksize: int = None
    _buffersize: int = None
    _dtype: _np.dtype = None
    _channels: Dict[str, List[int]] = None

    _instance = None

    def __init__(self):
        """
        This class is a singleton, which means that any "new" instance is actually the same, pointed to by a new name.

            >>> new_config = ossom.Configurations()
            >>> new_config is ossom.config
            True

        Basically it provides a one time setup interface to convinently create `Recorder`s and `Player`s with only default parameters.

            >>> rec1 = Recorder(samplerate=44100)
            >>> config.samplerate = 44100
            >>> rec2 = Recorder()
            >>> rec1.samplerate == rec2.samplerate
            True

        Returns
        -------
        None.

        """
        return

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def __repr__(self):
        s = f"Configurations(\n\tsamplerate={self.samplerate},\n\tblocksize={self.blocksize},\n\tbuffersize={self.buffersize},\n\tdtype={self.dtype},\n\tchannels={self.channels})"
        return s

    @property
    def samplerate(self) -> int:
        """Sample rate, or sampling frequency."""
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
        """Size of audio chunk read on calls to `next`."""
        return self._blocksize

    @blocksize.setter
    def blocksize(self, new: int):
        self._blocksize = int(new)
        return

    @property
    def buffersize(self) -> int:
        """Total size of the audio buffer held by `Recorder` or `Player` objects."""
        return self._buffersize

    @buffersize.setter
    def buffersize(self, new: int):
        if new < self.samplerate * 5:
            warnings.warn("Buffer may be too short.", ResourceWarning)
        self._buffersize = int(new)
        return

    @property
    def dtype(self) -> _np.dtype:
        """The audio data type."""
        return self._dtype

    @dtype.setter
    def dtype(self, new: str or _np.dtype):
        new = _np.dtype(new).name
        if new == 'float64':
            new = 'float32'
        self._dtype = _np.dtype(new)
        return

    @property
    def channels(self) -> Dict[str, List[int]]:
        """
        Dictionary containing two lists that represents the device channels to be used.
        Zero indexed.
        """
        return self._channels

    @property
    def inChannels(self) -> List[int]:
        """List of requested device input channels. Zero indexed."""
        return self._channels['in']

    @inChannels.setter
    def inChannels(self, new: List[int]):
        if not isinstance(new, list):
            if not isinstance(new, int):
                raise ValueError("The input channels must be an integer or a list of integers representing the device channels. Zero indexed.")
            new = list(range(int))
        self._channels['in'] = new
        return

    @property
    def outChannels(self) -> List[int]:
        """List of requested device output channels. Zero indexed."""
        return self._channels['out']

    @outChannels.setter
    def outChannels(self, new: List[int]):
        if not isinstance(new, list):
            if not isinstance(new, int):
                raise ValueError("The output channels must be an integer or a list of integers representing the device channels. Zero indexed.")
            new = list(range(int))
        self._channels['in'] = new
        return

    def reset(self):
        """Set all configuration values back to module default."""
        self._samplerate: int = _default['samplerate']
        self._blocksize: int = _default['blocksize']
        self._buffersize: int = _default['buffersize']
        self._dtype: _np.dtype = _default['dtype']
        self._channels: List[int] = _default['channels']
        return

config = Configurations()
