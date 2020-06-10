#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""
import numpy as _np
import numba as _nb
import multiprocessing as _mp
import time
from ossom import Recorder, Player
from ossom.audio import AudioGenerator, AudioRingBuffer
from ossom.logger import Logger
from ossom.utils import max_abs, rms, dB


@_nb.njit
def _noise(gain: float, samplerate: int, tlen: float, nchannels: int) -> _np.ndarray:
    shape = (int(samplerate*tlen), nchannels)
    noise = _np.zeros(shape, dtype=_np.float32)
    noise[:] = _np.random.randn(*shape)
    noise[:, ] /= max_abs(noise)
    return gain*noise


def noise_gen(gain: float = 1/(2**0.5),
              samplerate: int = 48000,
              buffersize: int = 64,
              tlen: float = 5.0,
              nchannels: int = 2) -> AudioGenerator:
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
    return AudioGenerator(data, samplerate, buffersize)


def monitor_loop(arb: AudioRingBuffer, func: callable, running: _mp.Event):
    data = _np.zeros((5512, 1))
    dtime = round(data.shape[0] / arb.samplerate, 3)
    running.wait()
    nextTime = time.time() + dtime
    while running.is_set() or arb.ready2read:
        dif = round(nextTime - time.time(), 3)
        if dif > 0.:
            time.sleep(dif)
        if arb.ready2read < data.shape[0]:
            nr = arb.ready2read
        else:
            nr = data.shape[0]
        data[:nr] = arb.read_next(nr)
        data.fill(0)
        func(data)
        nextTime += dtime
    return


# def playrec():
#     proc = _mp.Process(name='SoundCard Monitor', target=monitor_loop, args=(arb, print_dB, runok))
#     with spk.player(ng.samplerate, ng.nchannels, 256) as p:
#         with mic.recorder(arb.samplerate, arb.nchannels, 256) as r:
#             proc.start()
#             aud = stream_loop(p, r, arb, runok)
#             proc.join()
#     proc.close()
#     return aud


ng = noise_gen()


# arb = AudioRingBuffer(samplerate=ng.samplerate, nsamples=ng.nsamples,
#                       nchannels=ng.nchannels, rbsize=ng.bufsize,
#                       dtype=ng.dtype)

# lgr = Logger()

# runok = _mp.Event()


r = Recorder()
p = Player()

p(ng), r()


# if __name__ == "__main__":
#     # aud = record()
#     aud = playrec()
#     # play()

#     # arb.close()
#     # arb.unlink()
