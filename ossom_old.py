#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:33:43 2020

@author: joaovitor
"""

import numpy as _np
# import soundfile as _sf
import soundcard as _sc
# from scipy import signal as _ss
# from matplotlib import pyplot as _plt
import multiprocessing as _mp
# from threading import Event
from typing import List, Union  # , Optional
from ringbuffer import AudioRingBuffer


def _max_abs(arr: _np.ndarray):
    return _np.max(_np.abs(arr))


def _noise(samplerate: int = 44100, tlen: float = 5.0, nchannels: int = 1):
    nois = _np.random.randn(int(samplerate*tlen), nchannels)
    for col in range(nois.shape[1]):
        nois[:, col] /= _max_abs(nois[:, col])
    return nois


def _sweep(f1: float = 2e1, f2: float = 2e4, tlen: float = 5.0, phi: float = 0.0,
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
            data (_np.ndarray): Audio data.
            channels (int): Total number of channels.
            numframes (int): Number of frames sent to audio buffer.

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

    def adapt(self, data: _np.ndarray, nframes: int, channels: int):
        """
        Adjust data for use as generator.

        Args:
            data (_np.ndarray): Audio data.
            nframes (int): Number of frames passed to audio buffer.
            channels (int): Total number of channels.

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

    def __init__(self, samplerate: int, blocksize: int, nframes: int,
                 channels: Union[int, List[int]], audiodata: _np.ndarray,
                 speakid: int = None, micid: int = None):
        """
        Run audio playing and recording.

        Args:
            samplerate (int): Number of samples per second.
            blocksize (int): Elements count in buffer.
            channels (Any[int, List[int]]): Total number of channels or channel mapping.
            audio (_np.ndarray): The audio data.

        Returns:
            None.

        """
        print("streamer init")
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.nframes = nframes
        print('attr done')
        self.spk = speakid if speakid is not None else _sc.default_speaker()
        self.mic = micid if micid is not None else _sc.default_microphone()
        print('got mic {self.mic} and speaker {self.spk}')
        self.audio = SignalGenerator(audiodata, self.channels, self.nframes)
        print("streamer end")
        return

    def loop(self):
        """
        Run the application in blocking mode.

        Returns:
            None.

        """
        print("enter loop")
        with self.mic.recorder(self.samplerate, channels=self.channels,
                               blocksize=self.blocksize) as r:
            with self.spk.player(self.samplerate, channels=self.channels,
                                 blocksize=self.blocksize) as p:
                print('enter context manager')
                rec = AudioRingBuffer(samplerate=self.samplerate,
                                      nsamples=self.audio.nframes*self.audio.nblocks,
                                      nchannels=self.channels,
                                      rbsize=self.nframes)
                while True:
                    try:
                        p.play(next(self.audio))
                        rec.write_next(r.record(self.nframes))
                    except StopIteration:
                        break
        recdata = rec.get_audio()
        return recdata

    def ploop(self):
        """
        Run loop in parallel and waits for it to finish.

        Returns:
            None.

        """
        proc = _mp.Process(target=self.loop)
        proc.start()
        proc.join()
        return


def paral_streaming(samplerate: int, blocksize: int,
                    channels: Union[int, List[int]], audio: _np.ndarray):
    """
    Alternative function to test multiprocessing and soundcard.

    Instantiates the streamer already inside the parallel process.

    Args:
        samplerate (int): Number of samples per second.
        blocksize (int): Elements count in buffer.
        channels (Any[int, List[int]]): Total number of channels or channel mapping.
        audio (_np.ndarray): The audio data.

    Returns:
        None.

    """
    stream = Streamer(samplerate, blocksize, 128, channels, audio)
    stream.loop()
    return


SAMPLE_RATE = 44100
BLOCK_SIZE = 512
CHANNELS = 1
TOTAL_TIME = 5
TOTAL_SAMPLES = TOTAL_TIME * SAMPLE_RATE
NUM_BLOCKS = int(_np.ceil(TOTAL_SAMPLES/BLOCK_SIZE))


if __name__ == "__main__":
    import sys
    try:
        aud = sys.argv[1]
    except IndexError:
        aud = 'sweep'

    if aud == 'noise':
        AUDIO = _noise(tlen=TOTAL_TIME)
    elif aud == 'sweep':
        AUDIO = _sweep(tlen=TOTAL_TIME)
    else:
        raise ValueError

    stream = Streamer(SAMPLE_RATE, BLOCK_SIZE, 128, CHANNELS, AUDIO)
    rec = stream.loop()

    print("THE END!")
