#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""


import numpy as _np
import numba as _nb
# import multiprocessing as _mp
from ossom import Recorder, Player, Audio, LogMonitor, config
from ossom.utils import max_abs


@_nb.njit
def _noise(gain: float, samplerate: int, tlen: float, nchannels: int) -> _np.ndarray:
    shape = (int(samplerate*tlen), nchannels)
    noise = _np.zeros(shape, dtype=_np.float32)
    noise[:] = _np.random.randn(*shape)
    noise[:, ] /= max_abs(noise)
    grms = 10**(gain/20)
    return grms*noise


def noise(gain: float = -6,
          samplerate: int = 48000,
          buffersize: int = 64,
          tlen: float = 5.0,
          nchannels: int = 2) -> Audio:
    """
    Create an AudioGenerator of random noise.

    Parameters
    ----------
    gain : float, optional
        DESCRIPTION. The default is 1/(2**0.5).
    samplerate : int, optional
        DESCRIPTION. The default is 44100.
    buffersize : int, optional
        DESCRIPTION. The default is 64.
    tlen : float, optional
        DESCRIPTION. The default is 5.0.
    nchannels : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    AudioGenerator
        DESCRIPTION.

    """
    data = _noise(gain, samplerate, tlen, nchannels)
    return Audio(data, samplerate, buffersize)


def retrieve_device_ids():
    default = config.device
    print(config.list_devices())
    print("\nEscolha os dispositivos separados por vírgula")
    try:
        return [int(dev.strip()) for dev in input("No formato IN, OUT: ").split(",")]
    except ValueError:
        return default


if __name__ == "__main__":
    # Generate an Audio object of a random white noise
    ng = noise()

    # Prepare a monitoring object that stores in a file logger.
    lgr = LogMonitor()

    # Select default devices.
    config.device = retrieve_device_ids()

    # Create a recorder object to capture audio data.
    r = Recorder()

    # Create a player object to playback audio data.
    p = Player()

    # Tells logger which streamer object (player or recorder) to watch and how
    # many samples to read at each iteration.
    lgr(r, r.samplerate//8)

    # Plays the white noise
    p(ng)
    # While capturing audio
    r(ng.duration)
    # And asks logger to block until recorder has finished
    lgr.join()

    # Retrieves a copy of the recorded audio.
    a = r.get_record()
    # Playback the audio.
    p(a, blocking=True)

    # Delete both streamer objects for memory cleanup.
    del r, p
