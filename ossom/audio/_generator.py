# -*- coding: utf-8 -*-
"""Audio Generator."""

import numpy as np
from audio import Audio
from multiprocessing import Event


class AudioGenerator(Audio):
    """Provides interface for generating audio sinals continuously or time based."""

    def __init__(self, data: np.ndarray, samplerate: int,
                 buffersize: int) -> None:
        """Register sampling rate, size of audio buffer and data type."""
        Audio.__init__(self, data, samplerate)
        self._bufsize = buffersize
        self.counter = int()
        self.stopStreamFlag = Event()
        self.stopStreamFlag.clear()
        return

    def __iter__(self):
        """Iterate method."""
        return self

    def __next__(self):
        """Return data from `counter` to `counter` + `bufsize`."""
        if self.counter > self.nsamples:
            self.stopStreamFlag.set()
            raise StopIteration
        data = self.data[self.counter:self.counter+self.bufsize]
        self.counter += self.bufsize
        return data

    @property
    def bufsize(self) -> int:
        """Amount of samples read on each call to `next()`."""
        return self._bufsize
