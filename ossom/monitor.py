# -*- coding: utf-8 -*-
"""
Lacks documentation!

Created on Sat Jun 27 20:25:33 2020

@author: João Vitor Gutkoski Paes
"""

import numpy as np
import time
import multiprocessing as mp
from ossom import Recorder, Player, Configurations
from typing import Union


config = Configurations()


class Monitor(object):
    """Monitor class."""

    def __init__(self,
                 target: callable = lambda x: x,
                 samplerate: int = config.samplerate,
                 waittime: float = 1.,
                 args: tuple = (0,)):
        """
        Control a multiprocessing.Process to visualize data from recording or playing.

        Parameters
        ----------
        target : callable, optional
            DESCRIPTION. The default is None.
        args : tuple, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.target = target
        self.waitTime = waittime
        self.samplerate = samplerate
        self.readLen = int(np.ceil(waittime * samplerate))
        self.args = args
        return

    def __call__(self, strm: Union[Recorder, Player] = None, blocksize: int = None):
        """
        Configure the monitor and starts de process.

        Parameters
        ----------
        strm : Union[Recorder, Player], optional
            DESCRIPTION. The default is None.
        buffersize : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.running = strm.running
        self.finished = strm.finished
        self._buffer = strm.get_buffer(blocksize=self.readLen)
        # assert self.buffer.data is strm.data
        self._process = mp.Process(target=self._loop)
        return

    def setup(self):
        """Any setup step needed to the end monitoring object. Must be overriden on subclasses."""
        pass

    def tear_down(self):
        """Any destroying step needed to finish the end monitoring object. Must be overriden on subclasses."""
        pass

    def _loop(self):
        """
        Actual monitoring loop.

        Parameters
        ----------
        buffer : Audio
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.setup()
        self.running.wait()
        time.sleep(0.25)
        self.nextTime = time.time() + self.waitTime
        while self.running.is_set():
            sleepTime = self.nextTime - time.time()
            if sleepTime > 0.:
                time.sleep(sleepTime)
            self.target(self._buffer.read_next(self.readLen), *self.args)
            self.nextTime = time.time() + self.waitTime
            if self.finished.is_set():
                break
        self.tear_down()
        return

    def start(self):
        """Start the parallel process."""
        self._process.start()
        return

    def wait(self):
        """Finish the parallel process."""
        self._process.join()
        self._process.close()
        # self._buffer.close()
        return
