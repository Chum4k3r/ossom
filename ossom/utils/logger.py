# -*- coding: utf-8 -*-
"""
This module provide two classes: `Now` and `Logger`.

The first is just a wrap around time.localtime() with human readable interface of the exact time instant the object is created.
The latter is a simple file logger that uses `Now` instances to record information.

Created on Sat Jun  6 00:39:46 2020

@author: joaovitor
"""

# import json
import time
import random as rd
from typing import TextIO


class Now(object):
    """Human readable data about date and time."""
    _keys = ('year', 'month', 'day', 'wday', 'hour', 'min', 'sec')
    _wdays = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}

    def __init__(self):
        localtime = time.localtime()
        self._hour = f"0{localtime.tm_hour}" if localtime.tm_hour < 10 else f"{localtime.tm_hour}"
        self._min = f"0{localtime.tm_min}" if localtime.tm_min < 10 else f"{localtime.tm_min}"
        self._sec = f"0{localtime.tm_sec}" if localtime.tm_sec < 10 else f"{localtime.tm_sec}"
        self._day = f"0{localtime.tm_mday}" if localtime.tm_mday < 10 else f"{localtime.tm_mday}"
        self._month = f"0{localtime.tm_mon}" if localtime.tm_mon < 10 else f"{localtime.tm_mon}"
        self._year = f"{localtime.tm_year}"
        self._wday = f"{Now._wdays[localtime.tm_wday]}"
        self._GMT = f"GMT_{localtime.tm_zone}"
        return

    def __str__(self):
        return f"{self.wday}_{self.date}_{self.time}_{self.GMT}"

    @property
    def date(self):
        return f"{self.day}-{self.month}-{self.year}"

    @property
    def time(self):
        return f"{self.hour}-{self.min}-{self.sec}"

    @property
    def hour(self):
        """Wall clock hours."""
        return self._hour

    @property
    def min(self):
        """Wall clock minutes."""
        return self._min

    @property
    def sec(self):
        """Wall clock seconds."""
        return self._sec

    @property
    def year(self):
        """Year."""
        return self._year

    @property
    def month(self):
        """Month."""
        return self._month

    @property
    def day(self):
        """Month day."""
        return self._day

    @property
    def wday(self):
        """Week day."""
        return self._wday

    @property
    def GMT(self):
        """Time zone."""
        return self._GMT


class Logger(object):
    """Simple logger."""

    _randlogs = ['Albatroz', 'Jaguatirica', 'Jibóia', 'Arara', 'Mico', 'Boitatá', 'Boto Cor-de-Rosa']

    def __init__(self, name: str = 'common', ext: str = 'log',
                 title: str = 'title', logend: str = 'end.') -> None:
        """
        File based logger.

        Parameters
        ----------
        name : str, optional
            File name. The default is 'common'.
        ext : str, optional
            File extension. The default is 'log'.
        title : str, optional
            A title or description for the log. The default is 'title'.
        logend : str, optional
            A tag that represents the end of a logging session.
            The default is 'end.'.

        Returns
        -------
        None

        """
        self._name = name
        self._ext = ext
        self._title = title
        self._logend = f"\n# {logend}\n{13*'-'}\n\n"
        self._time = Now()
        self._header = f'{13*"-"}\n# LOG OUTPUT FILE\n# {self.name}\n# {self.time}\n\n# {self.title}\n'
        self.fopen()
        return

    def __del__(self):
        """Close underlying file on del statement."""
        try:
            self.fclose()
        except AttributeError:
            pass
        return

    @property
    def name(self) -> str:
        """Log file name."""
        return self._name

    @property
    def ext(self) -> str:
        """File extension."""
        return self._ext

    @property
    def title(self) -> str:
        """Log title."""
        return self._title

    @property
    def time(self) -> str:
        """Time at log creation."""
        return str(self._time)

    @property
    def header(self) -> str:
        """File header."""
        return self._header

    @property
    def logend(self) -> str:
        """End of logging."""
        return self._logend

    @property
    def file(self) -> TextIO:
        """File pointer."""
        return self._file

    def fopen(self):
        """Open the log file, enabling `read` and `write`."""
        self._file = open(f'{self.name}.{self.ext}', mode='a+')
        return

    def start_log(self):
        """Write header to log file."""
        self.file.write(self.header)
        return

    def log(self, message: str = None) -> int:
        """
        Write `message` to the log file along with the current time.

        Parameters
        ----------
        message : str, optional
            The content of the logging. If no input is given, randomizes an item from `LogFile._randlog`.
            The default is None.

        Returns
        -------
        b : int
            The amount of bytes written to log file.

        """
        time = Now().time
        msg = message if not message is None \
            else Logger._randlogs[rd.randint(0, len(Logger._randlogs)-1)]
        return self.file.write(f"{time} : {msg}\n")

    def end_log(self):
        """Write the end of logging tag."""
        self.file.write(self.logend)
        return

    def fclose(self):
        """Close the log file."""
        self.file.close()
        del self._file
        return

    def print_log(self):
        """Print the log file content. File must be opened."""
        try:
            self.file.seek(0)
            print(self.file.read())
        except AttributeError:
            self.fopen()
            self.file.seek(0)
            print(self.file.read())
            self.fclose()
        return


def _random_log(logger, timeout):
    init = time.time()
    while (init + timeout) > time.time():
        time.sleep(0.125)
        logger.log()
    logger.end_log()
    return


if __name__ == '__main__':
    NAME = 'log'
    EXT = 'txt'
    FILENAME = f'{NAME}.{EXT}'
    TITLE = 'Overall Level'

    logger = Logger(NAME, EXT, TITLE)
    _random_log(logger, 2.)
    logger.fclose()

    logger.reopen()
    _random_log(logger, 3.)
    logger.fclose()
