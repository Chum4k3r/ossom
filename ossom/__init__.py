# -*- coding: utf-8 -*-
"""
Ossom
=====

Audio IO with easy access to memory buffers for real-time analysis and processing of data.
"""

import numpy as _np
import sounddevice as _sd

config = _sd.default
config.samplerate = 48000
config.channels = 2, 2
config.latency = 'low', 'low'
config.blocksize = 256
config.dtype = _np.float32().dtype
config.__dict__['list_devices'] = _sd.query_devices

from .audio import Audio, AudioBuffer
from .streamer import Recorder, Player
from .logger import Logger
# from .config import config
# from . import wavefile
from . import utils

