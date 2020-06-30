# -*- coding: utf-8 -*-
"""
Utilities module.

Created on Tue May  5 00:31:42 2020

@author: João Vitor Gutkoski Paes
"""

from .colore import ColorStr, colorir, pinta_texto, pinta_fundo
from .freq import freq_to_band, fractional_octave_frequencies, normalize_frequencies, freqs_to_center_and_edges
from .maths import max_abs, rms, dB
from .logger import Logger, Now

__all__ = [
    # colore
    'ColorStr',
    'colorir',
    'pinta_texto',
    'pinta_fundo',

    # freq
    'freq_to_band',
    'fractional_octave_frequencies',
    'freqs_to_center_and_edges',
    'normalize_frequencies',

    # maths
    'max_abs',
    'rms',
    'dB',

    # logger
    'Logger',
    'Now']
