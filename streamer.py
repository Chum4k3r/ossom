#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:37:06 2020

@author: joaovitor
"""

import numpy as _np
import sounddevice as _sd
import threading
from audio import Audio

class _CallbackContext(object):
    """Base class for specific stream callbacks."""

    blocksize = None
    data = None
    frame = 0
    input_channels = output_channels = None
    input_dtype = output_dtype = None
    input_mapping = output_mapping = None
    silent_channels = None

    def __init__(self, loop=False):
        """This is a modified copy of sounddevice._CallbackContext."""
        self.loop = loop
        self.event = threading.Event()
        self.status = _sd.CallbackFlags()
        return

    def check_data(self, data, mapping, device):
        """Check data and output mapping."""
        data = _np.asarray(data)
        if data.ndim < 2:
            data = data.reshape(-1, 1)
        frames, channels = data.shape
        dtype = _sd._check_dtype(data.dtype)
        mapping_is_explicit = mapping is not None
        mapping, channels = _sd._check_mapping(mapping, channels)
        if data.shape[1] == 1:
            pass  # No problem, mono data is duplicated into arbitrary channels
        elif data.shape[1] != len(mapping):
            raise ValueError(
                'number of output channels != size of output mapping')
        # Apparently, some PortAudio host APIs duplicate mono streams to the
        # first two channels, which is unexpected when specifying mapping=[1].
        # In this case, we play silence on the second channel, but only if the
        # device actually supports a second channel:
        if (mapping_is_explicit and _np.array_equal(mapping, [0]) and
                _sd.query_devices(device, 'output')['max_output_channels'] >= 2):
            channels = 2
        silent_channels = _np.setdiff1d(_np.arange(channels), mapping)
        if len(mapping) + len(silent_channels) != channels:
            raise ValueError('each channel may only appear once in mapping')

        self.data = data
        self.output_channels = channels
        self.output_dtype = dtype
        self.output_mapping = mapping
        self.silent_channels = silent_channels
        return frames

    def check_out(self, out, frames, channels, dtype, mapping):
        """Check out, frames, channels, dtype and input mapping."""
        import numpy as np
        if out is None:
            if frames is None:
                raise TypeError('frames must be specified')
            if channels is None:
                channels = _sd.default.channels['input']
            if channels is None:
                if mapping is None:
                    raise TypeError(
                        'Unable to determine number of input channels')
                else:
                    channels = len(np.atleast_1d(mapping))
            if dtype is None:
                dtype = _sd.default.dtype['input']
            out = np.empty((frames, channels), dtype, order='C')
        else:
            frames, channels = out.shape
            dtype = out.dtype
        dtype = _sd._check_dtype(dtype)
        mapping, channels = _sd._check_mapping(mapping, channels)
        if out.shape[1] != len(mapping):
            raise ValueError(
                'number of input channels != size of input mapping')

        self.out = out
        self.input_channels = channels
        self.input_dtype = dtype
        self.input_mapping = mapping
        return frames

    def callback_enter(self, status, data):
        """Check status and blocksize."""
        self.status |= status
        self.blocksize = min(self.frames - self.frame, len(data))
        return

    def read_indata(self, indata):
        # We manually iterate over each channel in mapping because
        # numpy.take(..., out=...) has a bug:
        # https://github.com/numpy/numpy/pull/4246.
        # Note: using indata[:blocksize, mapping] (a.k.a. 'fancy' indexing)
        # would create unwanted copies (and probably memory allocations).
        for target, source in enumerate(self.input_mapping):
            # If out.dtype is 'float64', 'float32' data is "upgraded" here:
            self.out[self.frame:self.frame + self.blocksize, target] = \
                indata[:self.blocksize, source]
        return

    def write_outdata(self, outdata):
        # 'float64' data is cast to 'float32' here:
        outdata[:self.blocksize, self.output_mapping] = \
            self.data[self.frame:self.frame + self.blocksize]
        outdata[:self.blocksize, self.silent_channels] = 0
        if self.loop and self.blocksize < len(outdata):
            self.frame = 0
            outdata = outdata[self.blocksize:]
            self.blocksize = min(self.frames, len(outdata))
            self.write_outdata(outdata)
        else:
            outdata[self.blocksize:] = 0
        return

    def callback_exit(self):
        if not self.blocksize:
            raise _sd.CallbackAbort
        self.frame += self.blocksize
        return

    def finished_callback(self):
        self.event.set()
        return

    def start_stream(self, StreamClass, samplerate, channels, dtype, callback,
                     blocking, **kwargs):
        # stop()  # Stop previous playback/recording
        self.stream = StreamClass(samplerate=samplerate,
                                  channels=channels,
                                  dtype=dtype,
                                  callback=callback,
                                  finished_callback=self.finished_callback,
                                  **kwargs)
        self.stream.start()
        # global _last_callback
        # _last_callback = self
        if blocking:
            self.wait()
        return

    def wait(self, ignore_errors=True):
        """Wait for finished_callback.

        Can be interrupted with a KeyboardInterrupt.

        """
        try:
            self.event.wait()
        finally:
            self.stream.close(ignore_errors)
        return self.status if self.status else None


class Recorder(_CallbackContext, Audio):
    """Audio recorder object."""

    def __init__(self, nsamples, samplerate, channelmap, dtype, blocking, **kwargs):
        _CallbackContext.__init__(self)
        self.nsamples = self.check_out(None, nsamples, None, dtype, channelmap)
        Audio.__init__(self.out, samplerate)
        return

    def start(self):
        self.start_stream(_sd.InputStream, self.samplerate)

