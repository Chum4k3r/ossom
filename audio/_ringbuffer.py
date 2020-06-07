# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:43:33 2020.

@author: joaovitor
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from audio import Audio


class AudioRingBuffer(Audio, sm.SharedMemory):
    """Dados de áudio em memória compartilhada."""

    def __init__(self, name: str = None, samplerate: int = 44100,
                 nsamples: int = 32, nchannels: int = 1,
                 rbsize: int = 16, dtype: np.dtype = np.float32) -> None:
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
        rbsize : int, optional
            Amount of samples to read on `next_read` calls. The default is 16.
        dtype : np.dtype, optional
            Sample data type. The default is np.float32.

        Returns
        -------
        None.

        """
        if name is None:
            sz = dtype.itemsize * nsamples * nchannels
            sm.SharedMemory.__init__(self, create=True, size=sz)
        else:
            sm.SharedMemory.__init__(self, name)
        buffer = np.ndarray((nsamples, nchannels),
                            dtype=dtype, buffer=self.buf)
        Audio.__init__(self, buffer, samplerate)
        self._windex = mp.Value('i', int())
        self._rindex = mp.Value('i', int())
        self._rbsize = rbsize
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
        return self._windex.value

    @widx.setter
    def widx(self, n):
        self._windex.value = n
        if self._windex.value == self.nsamples:
            self.full.set()
        elif self._windex.value > self.nsamples:
            raise StopIteration
        return

    @property
    def ridx(self) -> int:
        """Read data index."""
        return self._rindex.value

    @ridx.setter
    def ridx(self, n):
        self._rindex.value = n
        return

    @property
    def rbsize(self) -> int:
        """Read data index."""
        return self._rbsize

    @property
    def ready2read(self) -> int or None:
        """
        How many samples are ready to read.

        If the write index is smaller than read index returns None

        Returns
        -------
        int or None
            DESCRIPTION.

        """
        dif = self.widx - self.ridx
        return dif if dif > 0 else 0

    @property
    def is_full(self) -> bool:
        """Check if ringbuffer is full or not."""
        return self._full.is_set()

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
        wdata = self._write_check(data)
        self._data[self.widx:(self.widx+wdata.shape[0])] = wdata[:]
        self.widx += wdata.shape[0]
        return

    def read_next(self, nread: int = None) -> np.ndarray:
        """
        Read the next chunk of data from buffer.

        The amount of data read each time is defined by `rbsize`.
        Alternatively accepts `nread` to read an specific amount of samples.
        To check how many samples can read, check `ready2read`.

        Parameters
        ----------
        nread : int, optional
            Number of samples to read. The default is None.

        Returns
        -------
        data : np.ndarray
            The chunk of data as a numpy array.

        """
        nread = self._read_check(nread)
        data = self.data[self.ridx:self.ridx+nread]
        self.ridx += nread
        return data

    def get_audio(self) -> Audio:
        """
        Extract a copy of the buffer data as an Audio object.

        Returns
        -------
        Audio
            The buffer as an audio.

        """
        return Audio(self.data.copy(), self.samplerate)

    def _read_check(self, nread: int):
        check = False
        if self.ridx >= self.nsamples:
            raise StopIteration
        elif nread is None:
            check = True
        return self.rbsize if check else nread

    def _write_check(self, data: np.ndarray):
        check = False
        if data.shape[0] > (left := self.nsamples - self.widx):
            check = True
        return data[:left] if check else data
