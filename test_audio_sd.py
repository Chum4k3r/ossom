#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""
import numpy as _np
import numba as _nb
import sounddevice as _sd
import multiprocessing as _mp
import time
from audio import Audio, AudioGenerator, AudioRingBuffer
from logger import Logger


# def _max_abs(arr: _np.ndarray) -> _np.ndarray:
#     return _np.max(_np.abs(arr), axis=0)


@_nb.njit(parallel=True)
def _max_abs(arr: _np.ndarray) -> _np.ndarray:
    ma = _np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    for col in _nb.prange(ma.shape[1]):
        ma[:, col] = _np.max(_np.abs(arr[:, col]))
    return ma

@_nb.njit
def _noise(gain: float, samplerate: int, tlen: float, nchannels: int) -> _np.ndarray:
    shape = (int(samplerate*tlen), nchannels)
    noise = _np.zeros(shape, dtype=_np.float32)
    noise[:] = _np.random.randn(*shape)
    noise[:, ] /= _max_abs(noise)
    return gain*noise


def noise_gen(gain: float = 1/(2**0.5),
              samplerate: int = 44100,
              buffersize: int = 64,
              tlen: float = 5.0,
              nchannels: int = 1) -> AudioGenerator:
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


@_nb.njit(parallel=True)
def _rms(samples: _np.ndarray) -> _np.ndarray:
    rms = _np.zeros((1, samples.shape[1]), dtype=samples.dtype)
    for col in _nb.prange(rms.shape[1]):
        rms[:, col] = _np.mean(samples[:, col]**2)**0.5
    return rms


@_nb.njit
def _dB(samples: _np.ndarray) -> _np.ndarray:
    return 20*_np.log10(_rms(samples))


def print_rms(samples: _np.ndarray = None):
    """
    Print the root mean squared of audio array.

    Parameters
    ----------
    samples : _np.ndarray
        Array of audio data.

    Returns
    -------
    None.

    """
    print(f'{_rms(samples)=}')
    return


def print_dB(samples: _np.ndarray = None):
    """
    Print the full scale level of audio array.

    Parameters
    ----------
    samples : _np.ndarray
        Array of audio data.

    Returns
    -------
    None.

    """
    print(f'{_dB(samples)=}')
    return

# def monitor_loop(arb: AudioRingBuffer, func: callable, running: _mp.Event):
#     data = _np.zeros((5512, 1))
#     dtime = round(data.shape[0] / arb.samplerate, 3)
#     running.wait()
#     nextTime = time.time() + dtime
#     while running.is_set() or arb.ready2read:
#         dif = round(nextTime - time.time(), 3)
#         if dif > 0.:
#             time.sleep(dif)
#         if arb.ready2read < data.shape[0]:
#             nr = arb.ready2read
#         else:
#             nr = data.shape[0]
#         data[:nr] = arb.read_next(nr)
#         data.fill(0)
#         func(data)
#         nextTime += dtime
#     return


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


arb = AudioRingBuffer(samplerate=ng.samplerate, nsamples=ng.nsamples,
                      nchannels=ng.nchannels, rbsize=ng.bufsize,
                      dtype=ng.dtype)

lgr = Logger()

runok = _mp.Event()


# if __name__ == "__main__":
#     # aud = record()
#     aud = playrec()
#     # play()

#     # arb.close()
#     # arb.unlink()
