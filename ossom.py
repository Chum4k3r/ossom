#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:33:43 2020

@author: joaovitor
"""

import numpy as _np
import soundfile as _sf
import soundcard as _sc
from scipy import signal as _ss
from matplotlib import pyplot as _plt
import multiprocessing as _mp
# from threading import Event
from typing import List


def _max_abs(arr: _np.ndarray):
    return _np.max(_np.abs(arr))


def _noise(gain: float = 0.707, samplerate: int = 44100, tlen: float = 5.0, nchannels: int = 1):
    nois = _np.random.randn(int(samplerate*tlen), nchannels)
    for col in range(nois.shape[1]):
        nois[:, col] /= _max_abs(nois[:, col])
    return nois


def _sweep(gain: float = 0.707, f1: float = 2e1, f2: float = 2e4, tlen: float = 5.0, phi: float = 0.0,
           fs: int = 44100, novak: bool = True):
    if novak:
        L = _np.round(f1 * tlen / _np.log(f2/f1)) / f1
    else:
        L = tlen/_np.log(f2/f1)
    # sweeprate = 1/L/_np.log(2)
    # nsamples = L*_np.log(f2/f1)*fs
    time = _np.arange(0, tlen, 1/fs)
    suip = _np.sin(2 * _np.pi * f1 * L * (_np.exp(time / L) - 1) + phi)
    return suip


class SignalGenerator(object):
    """Delivers signal to audio reproduction."""

    def __init__(self, data: _np.ndarray, channels: int, numframes: int):
        """
        Signal handling.

        Args:
            data (_np.ndarray): DESCRIPTION.
            channels (int): DESCRIPTION.
            numframes (int): DESCRIPTION.

        Returns:
            None.

        """
        self.counter = int()
        self.adapt(data, numframes, channels)

    def __iter__(self):
        """Iteration."""
        return self

    def __next__(self):
        """Count up to nblocks."""
        if self.counter >= self.nblocks:
            raise StopIteration
        frames = self.data[self.counter]
        self.counter += 1
        return frames

    def adapt(self, data, nframes, channels):
        """
        Adapt the data for use as a generator.

        Args:
            data (TYPE): DESCRIPTION.
            nframes (TYPE): DESCRIPTION.
            channels (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        totalsamp = data.size
        nblocks = int(_np.ceil(totalsamp/nframes))
        self.data = _np.resize(data, (nblocks, nframes, channels))
        return

    @property
    def nblocks(self):
        """Total number of blocks of data."""
        return self.data.shape[0]

    @property
    def nframes(self):
        """Total number of frames on each block."""
        return self.data.shape[1]

    @property
    def channels(self):
        """Total number of channels."""
        return self.data.shape[2]


class Streamer(object):
    """Runs the audio playing and recording."""

    _nframes = 128

    def __init__(self, samplerate, blocksize, channels, audiogen):
        """
        Run audio playback and record.

        Args:
            samplerate (TYPE): DESCRIPTION.
            blocksize (TYPE): DESCRIPTION.
            channels (TYPE): DESCRIPTION.
            audiogen (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.spk = _sc.default_speaker()
        self.mic = _sc.default_microphone()

        self.audio = SignalGenerator(audiogen, self.channels, self._nframes)

    def loop(self):
        """
        Run the application in blocking mode.

        Returns:
            None.

        """
        with self.spk.player(self.samplerate, channels=self.channels,
                        blocksize=self.blocksize) as p, \
            self.mic.recorder(self.samplerate, channels=self.channels,
                         blocksize=self.blocksize) as r:
            rec = []
            while True:
                try:
                    p.play(next(self.audio))
                    rec.append(r.record(self._nframes))
                except StopIteration:
                    break
        print(len(rec))
        return

    def ploop(self):
        """
        Run the application in parallel process.

        Returns:
            None.

        """
        proc = _mp.Process(target=self.loop)
        proc.start()
        proc.join()
        proc.close()
        return


SAMPLE_RATE = 44100
BLOCK_SIZE = 512
CHANNELS = 1
TOTAL_TIME = 4
# TOTAL_SAMPLES = TOTAL_TIME * SAMPLE_RATE
# NUM_BLOCKS = int(_np.ceil(TOTAL_SAMPLES/BLOCK_SIZE))


if __name__ == "__main__":
    import sys
    try:
        aud = sys.argv[1]
        gain = sys.argv[2]
    except IndexError:
        if not aud:
            aud = 'sweep'
        if not gain:
            gain = 0.5

    if aud == 'noise':
        audio = _noise(gain)
    elif aud == 'sweep':
        audio = _sweep(gain)
    else:
        raise ValueError

    stream = Streamer(SAMPLE_RATE, BLOCK_SIZE, CHANNELS, audio)
    stream.ploop()

    print("THE END!")
