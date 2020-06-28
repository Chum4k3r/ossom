# -*- coding: utf-8 -*-
"""
Monitoring objects.

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
        self.buffer = strm.get_buffer(buffersize)
        self.process = mp.Process(target=self.loop, args=(self.buffer, ))
        return

    def setup(self):
        """Any setup step needed to the end monitoring object. Must be overriden on subclasses."""
        pass

    def tear_down(self):
        """Any destroying step needed to finish the end monitoring object. Must be overriden on subclasses."""
        pass

    def loop(self, buffer: Audio):
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

    def start(self):
        """Start the parallel process."""
        self.process.start()
        return

    def wait(self):
        """Finish the parallel process."""
        self.process.join()
        self.process.close()
        return
