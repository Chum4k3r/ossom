# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:12:12 2020

@author: joaovitor
"""


import numpy as _np
import soundcard as _sc
import multiprocessing as _mp
import threading as _td
from ossom import Audio, AudioBuffer, config
from typing import List


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
        self._channels = channels
        self.running = _mp.Event()
        self.finished = _mp.Event()
        return

    def _loop_wrapper(self, blocking: bool):
        self.finished.clear()
        self.reset()
        self._thread = _td.Thread(target=self._loop)
        self._thread.start()
        if blocking:
            self.finished.wait()
            self.stop()
        return

    @property
    def channels(self):
        return self._channels


class Recorder(_Streamer):
    """Recorder class."""

    def __init__(self, id: int or str = None,
                 samplerate: int = config.samplerate,
                 blocksize: int = config.blocksize,
                 channels: List[int] = config.channels,
                 buffersize: int = config.buffersize,
                 dtype: _np.dtype = config.dtype,
                 loopback: bool = False):
        _Streamer.__init__(self, samplerate, blocksize, channels, buffersize, dtype)
        self._mic = _sc.default_microphone() if not id \
            else _sc.get_microphone(id, include_loopback=loopback)
        return

    def __call__(self, tlen: float = 5., blocking: bool = False):
        self.frames = int(_np.ceil(tlen * self.samplerate))
        if self.frames > self.nsamples:
            raise MemoryError("Requested recording time is greater than available space.")
        self._loop_wrapper(blocking)
        return

    def _loop(self):
        with self._mic.recorder(self.samplerate, self.channels, self.blocksize) as r:
            self.running.set()
            while self.widx < self.frames:
                self.write_next(r.record(self.blocksize//4))
                if self.finished.is_set() or self.is_full:
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
        self.widx = 0
        return

    def get_record(self, blocksize: int = None):
        return Audio(self.data[:self.frames].copy(), self.samplerate,
                     self.blocksize if not blocksize else blocksize)


class Player(_Streamer):
    def __init__(self, id: int or str = None,
                 samplerate: int = config.samplerate,
                 blocksize: int = config.blocksize,
                 channels: List[int] = config.channels,
                 buffersize: int = config.buffersize,
                 dtype: _np.dtype = config.dtype):
        _Streamer.__init__(self, samplerate, blocksize, channels, buffersize, dtype)
        self._spk = _sc.default_speaker() if not id \
            else _sc.get_speaker(id)
        return

    def __call__(self, audio: Audio, blocking: bool = False):
        self.frames = audio.nsamples
        if self.frames > self.nsamples:
            raise MemoryError("Requested playback time is greater than available space.")
        self.data[:self.frames] = audio[:]
        self._loop_wrapper(blocking)
        return

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
        return Audio(self.data[:self.frames], self.samplerate,
                     self.blocksize if not blocksize else blocksize)
