# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:07:04 2020.

@author: joaovitor
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as sm


class Audio(object):
    """Audio data interface."""

    def __init__(self, data: np.ndarray, samplerate: int, buffersize: int = 16) -> None:
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
        self._samplerate = int(samplerate)
        self._bufsize = buffersize
        self._data = data.reshape((-1, 1)) if data.ndim < 2 else data
        self._ridx = int()
        return

    def __getitem__(self, key):
        """Route AudioData getitem to numpy.ndarray getitem."""
        return np.ndarray.__getitem__(self.data, key)

    # def __setitem__(self, key, val):
    #     """Route AudioData setitem to numpy.ndarray setitem."""
    #     return np.ndarray.__setitem__(self.data, key, val)

    def __iter__(self):
        """Iterate method."""
        return self

    def __next__(self):
        """Return data from `counter` to `counter` + `bufsize`."""
        if self.ridx > self.nsamples:
            raise StopIteration
        data = self.data[self.ridx:self.ridx+self.bufsize]
        self.ridx += self.bufsize
        return data

    @property
    def ridx(self) -> int:
        """Read data index."""
        return self._ridx

    @ridx.setter
    def ridx(self, n):
        self._ridx = n
        return

    @property
    def bufsize(self) -> int:
        """Amount of samples read on each call to `next()`."""
        return self._bufsize

    @property
    def samplerate(self) -> int:
        """Sample rate of the audio."""
        return self._samplerate

    @property
    def data(self):
        """Audio data as a numpy.ndarray."""
        return self._data

    @property
    def nsamples(self) -> int:
        """Total number of data."""
        return self.data.shape[0]

    @property
    def nchannels(self) -> int:
        """Total number of channels."""
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """Total time duration."""
        return self.nsamples/self.samplerate

    @property
    def samplesize(self) -> int:
        """Size of one sample of audio."""
        return self.data.itemsize

    @property
    def dtype(self) -> np.dtype:
        """Type of the data."""
        return self.data.dtype

    @property
    def bytesize(self) -> int:
        """Size, in bytes, of whole array. Same as `samplesize * nsamples * nchannels`"""
        return self.data.nbytes


class AudioBuffer(Audio, sm.SharedMemory):
    """Dados de áudio em memória compartilhada."""

    def __init__(self, name: str = None, samplerate: int = 44100,
                 nsamples: int = 32, nchannels: int = 1,
                 buffersize: int = 16, dtype: np.dtype = np.float32) -> None:
        """
        Buffer object intended to read and write audio samples.

        Parameters
        ----------
        name : str, optional
            An existing AudioRingBuffer or SharedMemory name.
            The default is None.
        samplerate : int, optional
            Audio sampling rate. The default is 44100.
        nsamples : int, optional
            Total number of samples. The default is 32.
        nchannels : int, optional
            Total number of channels. The default is 1.
        buffersize : int, optional
            Amount of samples to read on `next` calls. The default is 16.
        dtype : np.dtype, optional
            Sample data type. The default is np.float32.

        Returns
        -------
        None.

        """
        sz = dtype.itemsize * nsamples * nchannels

        if name is not None:
            try:
                sm.SharedMemory.__init__(self, name)
            except FileNotFoundError:
                sm.SharedMemory.__init__(self, name, create=True, size=sz)
        else:
            sm.SharedMemory.__init__(self, create=True, size=sz)
        buffer = np.ndarray((nsamples, nchannels),
                            dtype=dtype, buffer=self.buf)
        Audio.__init__(self, buffer, samplerate, buffersize)
        self._widx = mp.Value('i', int())
        self._full = mp.Event()
        self._full.clear()
        return

    def __del__(self):
        """Guarantee that SharedMemory calls close and unlink."""
        self.close()
        self.unlink()
        return

    @property
    def widx(self) -> int:
        """Write data index."""
        return self._widx.value

    @widx.setter
    def widx(self, n):
        self._widx.value = n
        if self._widx.value >= self.nsamples:
            self._full.set()
        return

    @property
    def ready2read(self) -> int or None:
        """
        How many samples are ready to read.

        If the write index is smaller than read index returns None

        Returns
        -------
        int
            Amount of samples available to read.

        """
        dif = self.widx - self.ridx
        return dif if dif > 0 else 0

    @property
    def is_full(self) -> bool:
        """Check if ringbuffer is full or not."""
        return self._full.is_set()

    def clear(self):
        self.data[:] = 0
        return

    def get_Audio(self, copy: bool = False) -> Audio:
        """
        Audio object that points to shared memory buffer.

        Returns
        -------
        Audio
            The buffer as a simple Audio object.

        """
        return Audio(self.data.copy() if copy else self.data, self.samplerate)

    def write_next(self, data: np.ndarray) -> int or None:
        """
        Write data to buffer.

        If `widx` gets equal to `nsamples` the `full` Event is set.

        Parameters
        ----------
        data : np.ndarray
            Samples to write on buffer.

        Returns
        -------
        int
            Amount of written samples.

        """
        wdata, wsz = self._write_check(data)
        self._data[self.widx:(self.widx + wsz)] = wdata[:]
        self.widx += wsz
        return

    def _write_check(self, data: np.ndarray):
        left = self.nsamples - self.widx
        return (data[:left], left) if data.shape[0] > left else (data, data.shape[0])
