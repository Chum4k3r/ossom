#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""


import numpy as _np
import numba as _nb
# import multiprocessing as _mp
import time
from ossom import Recorder, Player, Audio, Logger, config
from ossom.utils import max_abs, rms, dB


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


if __name__ == "__main__":
    ng = noise()

    lgr = Logger()

    print(config.list_devices())

    print("\nEscolha os dispositivos separados por vírgula", end='')
    try:
        config.device = [int(dev.strip()) for dev in input("No formato IN, OUT: ").split(",")]
    except ValueError:
        pass

    r = Recorder()
    b = r.get_buffer(buffersize=r.samplerate//8)
    # direct access to Recorder buffer reading 0.125 s of audio on each call to next.

    p = Player()

    p(ng)
    r(ng.duration)

    while r.stream.active:
        time.sleep(0.125)
        d = next(b)
        RMS = rms(d)
        db = dB(RMS)
        lgr.log(f'Data shape={d.shape}\tRMS={RMS}\tdB={db}')
    lgr.end_log()
    lgr.fclose()

    a = r.get_record()  # Retrieves a copy of the recorded audio.
    p(a, blocking=True)

    del r
    del p

