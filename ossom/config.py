# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:01:29 2020

@author: joaovitor
"""

import numpy as _np
from sounddevice import default as config

config.samplerate = 48000
config.channels = 2, 2
config.latency = 'low', 'low'
config.blocksize = 256
config.dtype = _np.float32().dtype
