# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:08:22 2020

@author: João Vitor Gutkoski Paes
"""

import numpy as np
import wave as wv
from audiodata import AudioData


class WaveFile(object):
    """Wave file object."""

    def __init__(self, filename: str) -> None:
        self.name = filename
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            raise exc_value
        return

    def write(self, ad: AudioData) -> None:
        with wv.open(self.name, "wb") as wfile:
            wfile.setnchannels(ad.channels)
            wfile.setframerate(ad.samplerate)
            wfile.setnframes(ad.numsamples)
            wfile.setsampwidth(ad.samplesize)
            wfile.writeframes(np.float32(ad.data))
        return

    def read(self) -> AudioData:
        with wv.open(self.name, "rb") as rfile:
            numsamples = rfile.getnframes()
            samplerate = rfile.getframerate()
            databuff = rfile.readframes(numsamples)
            data = np.frombuffer(databuff, dtype=np.float16)
        ad = AudioData(data, samplerate)
        return ad


if __name__ == "__main__":
    SAMPLE_RATE = 44100  # [Hz] one second of data
    TOTAL_TIME = 2
    TOTAL_SAMPLES = TOTAL_TIME * SAMPLE_RATE
    CHANNELS = 1
    FILE_NAME = 'testfile.wav'

    #timesteps = np.linspace(0, TOTAL_TIME, TOTAL_SAMPLES)
    #sine = np.sin(2*np.pi*250*timesteps)

    noise = np.random.randn(TOTAL_SAMPLES, CHANNELS)
    noise /= np.max(np.abs(noise))

    audiow = AudioData(noise, SAMPLE_RATE)

    with Wave(FILE_NAME) as File:
        File.write(audiow)
        audior = File.read()

    print(audior.data.dtype, audiow.data.dtype)
