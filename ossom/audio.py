# -*- coding: utf-8 -*-
"""
These classes are designed to provide objects that will handle reading and writing of audio data as numpy arrays.

They are two: `Audio`, which is a read-only object, and `AudioBuffer`, that is a subclass
of `Audio` and `multiprocessing.shared_memory.SharedMemory`, thus providing a cross-processes data read and write functionality.

Created on Fri May 29 16:07:04 2020.

@author: João Vitor Gutkoski Paes
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from ossom import Configurations


config = Configurations()


class Audio(object):
    """Audio data interface."""

    def __init__(self, data: np.ndarray, samplerate: int,
                 blocksize: int = config.blocksize) -> None:
        """
        Audio objects are a representation of a waveform and a sample rate.

        They hold the basic information to play sound.

        Parameters
        ----------
        data : np.ndarray
            An array containing audio data, or a buffer to be filled with audio.
        samplerate : int
            Audio sample rate, or how many data represent one second of data.
        blocksize : int, optional
            Amount of samples to read on each call to `next`.
            The default is config.blocksize.

        Returns
        -------
        None

        """
        self._samplerate = int(samplerate)
        self._blocksize = blocksize
        self._data = data.reshape((-1, 1)) if data.ndim < 2 else data
        self._ridx = int()
        return

    def __getitem__(self, key):
        """Route Audio getitem to numpy.ndarray getitem."""
        return np.ndarray.__getitem__(self.data, key)

    def __iter__(self):
        """Iterate method."""
        return self

    def __next__(self) -> np.ndarray:
        """Return data from `ridx` to `ridx` + `blocksize`."""
        return self.read_next(self.blocksize)

    def read_next(self, blocksize: int) -> np.ndarray:
        """
        Read data from `ridx` to `ridx` + `blocksize`.

        Parameters
        ----------
        blocksize : int
            The amount of data to read.

        Raises
        ------
        StopIteration
            Unavailable to read more data than `nsamples`.

        Returns
        -------
        data : np.ndarray
            A numpy array with `blocksize` rows and `nchannels` columns.

        """
        if self.ridx > self.nsamples:
            raise StopIteration
        data = self.data[self.ridx:self.ridx+blocksize]
        self.ridx += blocksize
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
    def blocksize(self) -> int:
        """Amount of samples read on each call to `next()`."""
        return self._blocksize

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
        """Size, in bytes, of whole array. Same as `samplesize * nsamples * nchannels`."""
        return self.data.nbytes


class AudioBuffer(Audio, sm.SharedMemory):
    """Audio data in a shared memory buffer."""

    def __init__(self, name: str, samplerate: int,
                 buffersize: int, nchannels: int,
                 blocksize: int, dtype: np.dtype = config.dtype) -> None:
        """
        Buffer object intended to read and write audio samples.

        Parameters
        ----------
        name : str
            The SharedMemory name, can be None to automatically generate one.
        samplerate : int
            Audio sampling rate.
        buffersize : int
            Total number of samples.
        nchannels : int
            Total number of channels.
        blocksize : int
            Amount of samples to read on each call to `next`
        dtype : np.dtype
            Sample data type. The default is config.dtype.

        Returns
        -------
        None.

        """
        sz = dtype.itemsize * buffersize * nchannels

        if name is not None:
            try:
                sm.SharedMemory.__init__(self, name)
            except FileNotFoundError:
                sm.SharedMemory.__init__(self, name, create=True, size=sz)
        else:
            sm.SharedMemory.__init__(self, create=True, size=sz)
        buffer = np.ndarray((buffersize, nchannels),
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
        """Set all data to zero."""
        self.data[:] = 0
        return

    def get_audio(self, blocksize: int = None, copy: bool = False) -> Audio:
        """
        `Audio` object that points to shared memory buffer, or a copy of it.

        Parameters
        ----------
        blocksize : int, optional
            The read block size. The default is None.
        copy : bool, optional
            If set to `True`, the `Audio` is just a copy of the buffer. The default is False.

        Returns
        -------
        Audio
            The buffer as a read-only Audio object.

        """
        return Audio(self.data if not copy else self.data.copy(),
                     self.samplerate, blocksize=blocksize if blocksize else self.blocksize)

    def write_next(self, data: np.ndarray) -> int or None:
        """
        Write data to buffer.

        If `widx` gets equal to `nsamples` the buffer is set to full.
        Checking can be made using the `is_full` property.

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
