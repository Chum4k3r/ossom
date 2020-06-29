# -*- coding: utf-8 -*-
"""
Streaming objects.

Created on Sun Jun  7 15:37:06 2020

@author: joaovitor
"""

import numpy as _np
import sounddevice as _sd
import multiprocessing as mp
from ossom import config, Audio, AudioBuffer
from typing import List, Union, Tuple


class _Streamer(AudioBuffer):
    """Base streamer class."""

    def __init__(self, name: str, device: List[int or str], bufsize: int, samplerate: int,
                 channelmap: List[int], blocksize: int, dtype: _np.dtype, loop: bool):
        self.loop = loop
        self.finished = mp.Event()
        self.running = mp.Event()
        self.status = _sd.CallbackFlags()
        self.channels = channelmap.copy()
        self.device = device
        AudioBuffer.__init__(self, name, samplerate, bufsize, len(self.channels), blocksize, dtype)
        return

    def __del__(self):
        self.stream.stop()
        self.stream.close()
        self.close()
        self.unlink()
        return

    @property
    def device(self) -> int or str:
        """Audio device."""
        return self._device

    @device.setter
    def device(self, dev: Union[int, str, List[int], List[str]]):
        if type(dev) not in (list, tuple, int, str):
            raise TypeError("Device should be a number, or a string, a pair of numbers, or of strings")
        else:
            self._device = dev
        return

    @property
    def channels(self) -> List[int]:
        """Map of channels."""
        return self._channels

    @channels.setter
    def channels(self, chmap: Union[int, List[int], Tuple[int]]):
        if type(chmap) not in (_np.ndarray, list, tuple):
            raise TypeError("Channel map should be an iterable with the hardware active channels")
        else:
            self._channels = list(chmap).copy()
        return

    def get_stream(self, StreamClass, samplerate, channels, dtype, callback, **kwargs):
        # stop()  # Stop previous playback/recording
        self.stream = StreamClass(samplerate=samplerate,
                                  channels=channels,
                                  dtype=dtype,
                                  callback=callback,
                                  finished_callback=self.finished_callback,
                                  **kwargs)
        return

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
        for target, source in enumerate(self.channels):
            # If out.dtype is 'float64', 'float32' data is "upgraded" here:
            self._data[self.frame:self.frame + self.blocksize, target] = \
                indata[:self.blocksize, source - 1]  # channels starts on 1, indexes on 0
        return

    def write_outdata(self, outdata):
        # 'float64' data is cast to 'float32' here:
        outdata[:self.blocksize, self.output_mapping] = \
            self.outdata[self.frame:self.frame + self.blocksize]
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
            raise _sd.CallbackStop
        self.frame += self.blocksize
        return

    def finished_callback(self):    # TODO> close stream!
        self.running.clear()
        self.finished.set()
        self.reset()
        print(f"Exiting {self}.")
        return

    def start(self, blocking: bool = False):
        if not self.stream.stopped:
            self.stop()
        self.stream.start()
        self.running.set()
        self.finished.clear()
        if blocking:
            self.wait()
        return

    def wait(self, ignore_errors=True):
        """Wait for finished_callback.

        Can be interrupted with a KeyboardInterrupt.

        """
        try:
            self.finished.wait()
        finally:
            self.stop(ignore_errors)
        return self.status if self.status else None

    def stop(self, ignore_errors=True):
        """Stop streaming of audio."""
        self.stream.stop(ignore_errors)
        self.running.clear()
        return

    def callback(self):
        """Stream callback."""
        pass

    def reset(self):
        """Reset data counter(s)."""
        pass


class Recorder(_Streamer):
    """Audio recorder object."""

    def __init__(self, name: str = None,
                 device: List[int or str] = config.device[0],
                 samplerate: int = config.samplerate,
                 bufsize: int = 10*config.samplerate,
                 channelmap: List[int] = [1, 2],
                 blocksize: int = config.blocksize,
                 dtype: _np.dtype = config.dtype[0],
                 loop: bool = False):
        """
        Create an object capable of recording audio using its `__call__` method.

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
        _Streamer.__init__(self, name, device, bufsize, samplerate, channelmap, blocksize, dtype, loop)
        self.get_stream(_sd.InputStream, self.samplerate, self.nchannels,
                        self.dtype, self.callback, device=self.device,
                        latency=config.latency[0])
        self.reset()
        return

    def __call__(self, tlen: float = None, blocking: bool = False):
        """
        Start recording audio data during tlen seconds.

        Parameters
        ----------
        tlen : float, optional
            DESCRIPTION. The default is None.
        blocking : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.frames = int(_np.ceil(tlen*self.samplerate)) if tlen is not None else self.nsamples
        # self.frames = self.check_out(self.data, recsamples, self.nchannels, self.dtype, self.channels)
        self.start(blocking)
        return

    def callback(self, indata, frames, time, status):
        """Stream callback function."""
        assert len(indata) == frames
        self.callback_enter(status, indata)
        self.read_indata(indata)
        self.widx = self.frame
        self.callback_exit()
        return

    def reset(self):
        """Reset the write indexes."""
        self.widx = self.frame = 0
        return

    def get_record(self):
        """Return an Audio object with a copy of last recorded data."""
        return Audio(self.data[:self.frames].copy(), self.samplerate, self.bufsize)

    # def check_out(self, out, frames, channels, dtype, mapping):
    #     """Check out, frames, channels, dtype and input mapping."""
    #     # import numpy as np
    #     if channels is None:
    #         if mapping is None:
    #             raise TypeError(
    #                 'Unable to determine number of input channels')
    #         channels = _sd.default.channels['input']
    #     else:
    #         channels = len(_np.atleast_1d(mapping))
    #     if dtype is None:
    #         dtype = _sd.default.dtype['input']
    #     else:
    #         frames, channels = out.shape
    #         dtype = out.dtype
    #     dtype = _sd._check_dtype(dtype)
    #     mapping, channels = _sd._check_mapping(mapping, channels)
    #     if out.shape[1] != len(mapping):
    #         raise ValueError(
    #             'number of input channels != size of input mapping')

    #     # self.out = self.data  # Audio have data
    #     # self.frame = 0  # AudioBuffer have ridx
    #     # self.input_channels = channels  # Audio already have nchannels
    #     # self.input_dtype = dtype  # Audio already have dtype
    #     # self.input_mapping = mapping  # Streamer have channels wich are a mapping
    #     return frames


class Player(_Streamer):
    """Audio player object."""

    def __init__(self, name: str = None,
                 device: List[int or str] = config.device[0],
                 samplerate: int = config.samplerate,
                 bufsize: int = 10*config.samplerate,
                 channelmap: List[int] = [1, 2],
                 blocksize: int = config.blocksize,
                 dtype: _np.dtype = config.dtype[1],
                 loop: bool = False):
        _Streamer.__init__(self, name, device, bufsize, samplerate, channelmap, blocksize, dtype, loop)
        self.get_stream(_sd.OutputStream, self.samplerate, self.channels[-1],
                        self.dtype, self.callback,
                        prime_output_buffers_using_stream_callback=False,
                        latency=config.latency[1])
        self.reset()
        return

    def __call__(self, audio: Audio, blocking: bool = False):
        """Begin to play audio data."""
        # TODO: Checar se o objeto de áudio realmente é compativel com o reprodutor
        self.frames = self.check_data(audio.data, self.channels, self.device)
        self.start(blocking)
        return

    def callback(self, outdata, frames, time, status):
        """Stream callback function."""
        assert len(outdata) == frames
        self.callback_enter(status, outdata)
        self.write_outdata(outdata)
        self.ridx = self.frame
        self.callback_exit()
        return

    def reset(self):
        """Reset reading indexes to zero."""
        self.ridx = self.frame = 0
        return

    def check_data(self, data, mapping, device):
        """Check data and output mapping."""
        # data = _np.asarray(data)
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

        self.outdata = data
        # self.frame = 0
        self.output_channels = channels  # Audio have nchannels
        self.output_dtype = dtype  # Audio have dtype
        self.output_mapping = mapping  # Streamer have channels
        self.silent_channels = silent_channels
        return frames
