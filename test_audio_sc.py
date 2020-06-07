#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""

import numpy as _np
import soundcard as _sc
import multiprocessing as _mp
import time
from audio import Audio, AudioGenerator, AudioRingBuffer


def _max_abs(arr: _np.ndarray):
    return _np.max(_np.abs(arr))


def _noise(gain: float, samplerate: int, tlen: float, nchannels: int):
    noise = _np.random.randn(int(samplerate*tlen), nchannels)
    for col in range(noise.shape[1]):
        noise[:, col] /= _max_abs(noise[:, col])
    return gain*noise


def noise_gen(gain: float = 1/(2**0.5),
              samplerate: int = 44100,
              buffersize: int = 64,
              tlen: float = 5.0,
              nchannels: int = 1) -> AudioGenerator:
    """
    Create an audio generator of random noise.

    Parameters
    ----------
    samplerate : int, optional
        DESCRIPTION. The default is 44100.
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


def stream_loop(player, recorder, arb, running):
    runok.set()
    while True:
        try:
            arb.write_next(recorder.record(arb.rbsize)) if recorder is not None else None
            player.play(next(ng)) if player is not None else None
        except StopIteration:
            if arb.ready2read:
                continue
            runok.clear()
            break
        # print(f'from this process {arb.widx=}')
    aud = arb.get_audio() if recorder is not None else None
    return aud if recorder is not None else None



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
        print(f'{nr=}')
        data[:nr] = arb.read_next(nr)
        data.fill(0)
        func(data)
        nextTime += dtime
    return


def record():
    proc = _mp.Process(name='SoundCard Monitor', target=monitor_loop, args=(arb, print_dB, runok))
    with mic.recorder(arb.samplerate, arb.nchannels, 256) as r:
        proc.start()
        aud = stream_loop(None, r, arb, runok)
        proc.join()
    proc.close()
    return aud


def play():
    with spk.player(ng.samplerate, ng.nchannels, 256) as p:
        stream_loop(p, None, arb, runok)
    return


def playrec():
    proc = _mp.Process(name='SoundCard Monitor', target=monitor_loop, args=(arb, print_dB, runok))
    with spk.player(ng.samplerate, ng.nchannels, 256) as p:
        with mic.recorder(arb.samplerate, arb.nchannels, 256) as r:
            proc.start()
            aud = stream_loop(p, r, arb, runok)
            proc.join()
    proc.close()
    return aud



def dB(samples):
    return 20*_np.log10(_np.mean(samples**2)**0.5)

def print_dB(samples):
    print(f'{dB(samples)=}')
    return


ng = noise_gen()


arb = AudioRingBuffer(samplerate=ng.samplerate, nsamples=ng.nsamples,
                      nchannels=ng.nchannels, rbsize=ng.bufsize,
                      dtype=ng.dtype)


spks = _sc.all_speakers()
mics = _sc.all_microphones(include_loopback=True)


runok = _mp.Event()


if __name__ == "__main__":
    # aud = record()
    aud = playrec()
    # play()

    spk.play(aud.data, aud.samplerate)

    # arb.close()
    # arb.unlink()
