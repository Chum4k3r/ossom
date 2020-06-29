#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:27:43 2020.

@author: joaovitor
"""


import numpy as _np
import numba as _nb
from ossom import Recorder, Player, Audio, Logger, Monitor, config
from ossom.utils import max_abs, rms, dB


class LogMonitor(Logger, Monitor):
    """File logger based monitor."""

    def __init__(self, name: str = 'example', ext: str = 'log',
                 samplerate: int = config.samplerate, waittime: float = 0.125,
                 title: str = 'Example logging.', logend: str = 'D End.') -> None:
        Logger.__init__(self, name, ext, title, logend)
        self.fclose()  # On windows it fails if the file is open on process start.
        Monitor.__init__(self, self.do_logging, samplerate, waittime, tuple())
        return

    def setup(self):
        """Open log file."""
        self.fopen()
        self.start_log()
        return

    def do_logging(self, data):
        """Process and log."""
        RMS = rms(data)
        db = dB(RMS)
        self.log(f'Data shape={data.shape}\tRMS={RMS}\tdB={db}')
        return

    def tear_down(self):
        """End the log and close file."""
        self.end_log()
        self.fclose()
        return


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
          blocksize: int = 64,
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
    return Audio(data, samplerate, blocksize)


def retrieve_device_ids():
    """Select audio IO devices."""
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
    # config.device = retrieve_device_ids()

    # Create a recorder object to capture audio data.
    r = Recorder()

    # Create a player object to playback audio data.
    p = Player()

    # Tells logger which streamer object (player or recorder) to watch and how
    # many samples to read at each iteration.
    lgr(r, r.samplerate//8)

    # Start the monitor before the audio streamer
    lgr.start()
    # Plays the white noise
    p(ng)
    # While capturing audio
    r(ng.duration)

    # And asks logger to block until recorder has finished
    lgr.wait()

    # # Now set the logger to watch the player object
    lgr(p, p.samplerate//8)

    # # Retrieves a copy of the recorded audio.
    a = r.get_record()

    # # Start the monitor before the audio streamer
    lgr.start()
    # # Playback the audio.
    p(a)

    # # Finishes logger services.
    lgr.wait()

    # Delete objects for memory cleanup. Explicit deleting is necessary.
    # del r, p, lgr
