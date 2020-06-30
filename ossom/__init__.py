# -*- coding: utf-8 -*-
"""
Audio IO with easy access to memory buffers for real-time analysis and processing of data.

Author: João Vitor Gutkoski Paes
"""

from . import utils
from .configurations import Configurations
from .audio import Audio, AudioBuffer
from .streamer import Recorder, Player
from .monitor import Monitor

__all__ = ['Audio', 'AudioBuffer',
           'Recorder', 'Player',
           'Monitor',
           'Configurations',
           'utils']

