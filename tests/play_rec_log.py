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
    return gain*noise


def noise(gain: float = 1/(2**0.5),
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


ng = noise()

lgr = Logger()

print(config.list_devices())
# input()

r = Recorder()
a = r.get_Audio()

p = Player()

# assert hex(id(a.data)) == hex(id(r.data))  # both are the same memory space

p(ng)
r(ng.duration)

while r.stream.active:
    d = next(a)
    RMS = rms(d)
    db = dB(RMS)
    lgr.log(f'RMS={RMS}  dB={db}')
    time.sleep(0.125)
lgr.end_log()
lgr.fclose()

a = r.get_record()
p(a, blocking=True)

# if __name__ == "__main__":
#     # aud = record()
#     aud = playrec()
#     # play()

#     # arb.close()
#     # arb.unlink()
