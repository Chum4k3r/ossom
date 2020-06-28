# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:25:33 2020

@author: joaovitor
"""

# import numpy as np
import time
import multiprocessing as mp
from ossom import Recorder, Player, Audio
from typing import Union

class Monitor(object):
    """Monitor class."""

    def __init__(self,
                 target: callable = lambda x: x,
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
        self.args = args
        return

    def __call__(self, strm: Union[Recorder, Player] = None, buffersize: int = None):
        """
        Configures the monitor and starts de process.

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
        self.buffer = strm.get_buffer(buffersize, copy=False)
        self.process = mp.Process(target=self.loop, args=(self.buffer, ))
        self.process.start()
        return

    def setup(self):
        pass

    def tear_down(self):
        pass

    def loop(self, buffer: Audio):
        """
        The actual loop.

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
        self.nextTime = time.time() + self.waitTime
        while self.running.is_set():
            sleepTime = self.nextTime - time.time()
            if sleepTime > 0.:
                time.sleep(sleepTime)
            self.target(buffer, *self.args)
            if self.finished.is_set():
                break
            self.nextTime += self.waitTime
        self.tear_down()
        return

    def join(self):
        """
        Finish the parallel process.

        Returns
        -------
        None.

        """
        self.process.join()
        return