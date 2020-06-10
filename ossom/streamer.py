#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:37:06 2020

@author: joaovitor
"""

import numpy as _np
import sounddevice as _sd
import threading
from audio import AudioRingBuffer
from typing import List, Union


config = _sd.default
config.samplerate = 48000
config.channels = 2, 2
config.latency = 'low', 'low'
config.blocksize = 256
config.dtype = _np.float32().dtype


class _CallbackContext(object):  # (AudioRingBuffer):  # TODO: Fazer o contexto herdar AudioRingBuffer
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

        self.data[:] = data[:]
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


class _Streamer(AudioRingBuffer, _CallbackContext):
    """Base streamer class."""

    def __init__(self, device: List[int or str], bufsize: int, samplerate: int,
                 channelmap: List[int], blocksize: int, dtype: _np.dtype, loop: bool):
        _CallbackContext.__init__(self)
        self.channels = _np.array(channelmap, copy=True).tolist()
        self.device = device
        AudioRingBuffer.__init__(self, None, samplerate, bufsize, len(self.channels), blocksize, dtype)
        return

    @property
    def device(self) -> int or str:
        """The audio device."""
        return self._device

    @device.setter
    def device(self, dev: Union[int, str, List[int], List[str]]):
        if type(dev) not in (list, tuple, int, str):
            raise TypeError("Device should be a number, or a string, a pair of numbers, or of strings")
        else:
            self._device = dev
        return

    def stop(self, ignore_errors=True):
        """Stop streaming of audio."""
        self.stream.stop(ignore_errors)
        self.stream.close(ignore_errors)
        return

    def callback(self):
        """Stream callback."""
        pass

    def reset(self):
        """Reset data counter(s)."""
        pass


class Recorder(_Streamer):
    """Audio recorder object."""

    def __init__(self, device: List[int or str] = config.device[0], samplerate: int = config.samplerate,
                 bufsize: int = 5*config.samplerate, channelmap: List[int] = [1, 2],
                 blocksize: int = config.blocksize, dtype: _np.dtype = config.dtype[0], loop: bool = False):
        """
        Creates an object capable of recording audio using its `__call__` method.

            >>> r = Recorder()  # creates a recorder object
            >>> r()  # records data during 5 second by default
            >>> a = r.get_audio()  # retrieves recorded audio


        Parameters
        ----------
        device : List[int or str]
            Sound device.
        bufsize : int, optional
            Total size number of samples to allocate as memory. The default is 48000.
        samplerate : int, optional
            Sample rate. The default is 48000.
        channelmap : List[int], optional
            List of hardware channels to record, startin from 1. The default is [1, 2].
        blocksize : int, optional
            Latency. The default is 256.
        dtype : _np.dtype, optional
            Recorded array data type. The default is _np.float32().dtype.
        blocking : bool, optional
            If set to True, wait for recording before returning from call. The default is False.

        Returns
        -------
        None.

        """
        _Streamer.__init__(self, device, bufsize, samplerate, channelmap, blocksize, dtype, loop)
        return

    def __call__(self, tlen: float = None, blocking: bool = False):
        recsamples = int(_np.ceil(tlen*self.samplerate)) if tlen is not None else self.nsamples
        self.frames = self.check_out(self.data, recsamples, self.nchannels, self.data.dtype, self.channels)
        self.start_stream(_sd.InputStream, self.samplerate, self.input_channels,
                          self.input_dtype, self.callback, blocking, device=self.device, latency=config.latency)
        return

    def callback(self, indata, frames, time, status):
        assert len(indata) == frames
        self.callback_enter(status, indata)
        self.read_indata(indata)
        self.callback_exit()
        self.widx = self.frame
        if self.is_full:
            raise _sd.CallbackAbort
        return

    def reset(self):
        self.widx = self.frame = 0
        return


class Player(_Streamer):
    """Audio player object."""

    def __init__(self, device: List[int or str] = config.device[0], samplerate: int = config.samplerate,
                 bufsize: int = 5*config.samplerate, channelmap: List[int] = [1, 2],
                 blocksize: int = config.blocksize, dtype: _np.dtype = config.dtype[1], loop: bool = False):
        _Streamer.__init__(self, device, bufsize, samplerate, channelmap, blocksize, dtype, loop)
        return

    def __call__(self, audio, blocking: bool = False):
        # TODO: Checar se o objeto de áudio realmente é compativel com o reprodutor
        self.frames = self.check_data(audio.data, self.channels, self.device)
        self.start_stream(_sd.OutputStream, self.samplerate, self.output_channels,
                         self.output_dtype, self.callback, blocking,
                         prime_output_buffers_using_stream_callback=False, latency=config.latency)
        return

    def callback(self, outdata, frames, time, status):
        assert len(outdata) == frames
        self.callback_enter(status, outdata)
        self.write_outdata(outdata)
        self.ridx = self.frame
        self.callback_exit()
        return

    def reset(self):
        self.ridx = self.frame = 0
        return