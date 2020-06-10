#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio module.

@author: joaovitor
"""

from ._audio import Audio
from ._generator import AudioGenerator
from ._ringbuffer import AudioRingBuffer


__version__ = '0.1.0a'
__all__ = ['Audio', 'AudioGenerator', 'AudioRingBuffer']
