# -*- coding: utf-8 -*-
"""
Lacks documentation!

Created on Sun Jun 28 21:12:12 2020

@author: João Vitor Gutkoski Paes
"""


import numpy as _np
import soundcard as _sc
import multiprocessing as _mp
import threading as _td
from ossom import Audio, AudioBuffer, Configurations
from typing import List


config = Configurations()


class _Streamer(AudioBuffer):
    """Base streamer class."""

    def __init__(self,
                 samplerate: int,
                 blocksize: int,
                 channels: List[int],
                 buffersize: int,
                 dtype: _np.dtype):
        AudioBuffer.__init__(self, None, samplerate, buffersize,
                             len(channels), blocksize//2, dtype)
        self.running = _mp.Event()
        self.finished = _mp.Event()
        return

    def get_buffer(self, blocksize: int = None):
        pass

    def _loop_wrapper(self, blocking: bool):
        self.finished.clear()
        self.reset()
        self._thread = _td.Thread(target=self._loop)
        self._thread.start()
        if blocking:
            self.finished.wait()
            self.stop()
        return


class Recorder(_Streamer):
    """Recorder class."""

    def __init__(self, id: int or str = None,
                 samplerate: int = config.samplerate,
                 blocksize: int = config.blocksize,
                 channels: List[int] = config.channels['in'],
                 buffersize: int = config.buffersize,
                 dtype: _np.dtype = config.dtype,
                 loopback: bool = False):
        """
        Record audio from input device directly into shared memory.



        Parameters
        ----------
        id : int or str, optional
            DESCRIPTION. The default is None.
        samplerate : int, optional
            DESCRIPTION. The default is config.samplerate.
        blocksize : int, optional
            DESCRIPTION. The default is config.blocksize.
        channels : List[int], optional
            DESCRIPTION. The default is config.channels.
        buffersize : int, optional
            DESCRIPTION. The default is config.buffersize.
        dtype : _np.dtype, optional
            DESCRIPTION. The default is config.dtype.
        loopback : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        _Streamer.__init__(self, samplerate, blocksize, channels, buffersize, dtype)
        self._channels = channels
        self._mic = _sc.default_microphone() if not id \
            else _sc.get_microphone(id, include_loopback=loopback)
        return

    def __call__(self, tlen: float = 5., blocking: bool = False):
        self.frames = int(_np.ceil(tlen * self.samplerate))
        if self.frames > self.nsamples:
            raise MemoryError("Requested recording time is greater than available space.")
        self._loop_wrapper(blocking)
        return

    @property
    def channels(self):
        """The device channels to record from. Zero indexed."""
        return self._channels

    def _loop(self):
        with self._mic.recorder(self.samplerate, self.channels, self.blocksize) as r:
            self.running.set()
            while self.widx < self.frames:
                self.write_next(r.record(self.blocksize//4))
                if self.finished.is_set() or self.is_full:
                    break
            r.flush()
        self.running.clear()
        self.finished.set()
        return

    def stop(self):
        if not self.finished.is_set():
            self.finished.set()
        self._thread.join()
        return

    def reset(self):
        self.widx = 0
        return

    def get_record(self, blocksize: int = None):
        return Audio(self.data[:self.frames].copy(), self.samplerate,
                     self.blocksize if not blocksize else blocksize)


class Player(_Streamer):
    def __init__(self, id: int or str = None,
                 samplerate: int = config.samplerate,
                 blocksize: int = config.blocksize,
                 channels: List[int] = config.channels['out'],
                 buffersize: int = config.buffersize,
                 dtype: _np.dtype = config.dtype):
        _Streamer.__init__(self, samplerate, blocksize, channels, buffersize, dtype)
        self._channels = channels
        self._spk = _sc.default_speaker() if not id \
            else _sc.get_speaker(id)
        return

    def __call__(self, audio: Audio, blocking: bool = False):
        if self.nchannels != audio.nchannels:
            raise ValueError("The number of channels is incompatible.")
        self.frames = audio.nsamples
        if self.frames > self.nsamples:
            raise MemoryError("Requested playback time is greater than available space.")
        self.data[:self.frames] = audio[:]
        self._loop_wrapper(blocking)
        return

    @property
    def channels(self):
        """The device channels to output data to. Zero indexed."""
        return self._channels

    def _loop(self):
        with self._spk.player(self.samplerate, self.channels, self.blocksize) as p:
            self.running.set()
            while self.ridx < self.frames:
                p.play(self.read_next(self.blocksize//4))
                if self.finished.is_set():
                    break
        self.running.clear()
        self.finished.set()
        return

    def stop(self):
        if not self.finished.is_set():
            self.finished.set()
        self._thread.join()
        return

    def reset(self):
        self.ridx = 0
        return

    def get_playback(self, blocksize: int = None):
        return Audio(self.data[:self.frames].copy(), self.samplerate,
                     self.blocksize if not blocksize else blocksize)
