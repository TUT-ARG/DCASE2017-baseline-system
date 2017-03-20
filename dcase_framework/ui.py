#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interfacing
================

Utility classes for user interfacing.

FancyLogger
^^^^^^^^^^^

This is class to provide light extra formatting for system status logging.

.. autosummary::
    :toctree: generated/

    FancyLogger
    FancyLogger.title
    FancyLogger.section_header
    FancyLogger.foot
    FancyLogger.line
    FancyLogger.data
    FancyLogger.info
    FancyLogger.debug
    FancyLogger.error

"""

import logging


class FancyLogger(object):
    """Logger class
    """
    def __init__(self, logger=None):
        """Constructor

        Parameters
        ----------
        logger : object, optional
            Instance of logger class

        Returns
        -------

        nothing

        """

        self.logger = logger or logging.getLogger(__name__)

        self.separator = "=================================================="

    def title(self, text):
        """Title, logged at info level

        Parameters
        ----------
        text : str
            Title text

        Returns
        -------

        nothing

        """

        self.info(text)

    def section_header(self, text):
        """Section header, logged at info level

        Parameters
        ----------
        text : str
            Section header text

        Returns
        -------

        nothing

        """

        self.info(text)
        self.info(self.separator)
        self.info('')

    def foot(self, text='  DONE', time=None, item_count=None):
        """Footer, logged at info level

        Parameters
        ----------
        text : str, optional
            Footer text
        time : str, optional
            Elapsed time as string
        item_count : int, optional
            Item count

        Returns
        -------

        nothing

        """
        output = '{text:10s} '.format(text=text)

        if time:
            output += '[{time:<15s}] '.format(time=time)

        if item_count:
            output += '[{items:<d} items] '.format(items=item_count)

        self.info(output)
        self.info()

    def line(self, text='', level='info'):
        """Generic line logger
        Multiple lines are split and logged separately

        Parameters
        ----------
        text : str, optional
            Text
        level : str
            Logging level, one of [info,debug,warning,warn,error]
            Default value "info"

        Returns
        -------

        nothing

        """

        lines = text.split('\n')
        for line in lines:
            if level.lower() == 'info':
                self.logger.info(line)
            elif level.lower() == 'debug':
                self.logger.debug(line)
            elif level.lower() == 'warning' or level.lower() == 'warn':
                self.logger.warn(line)
            elif level.lower() == 'error':
                self.logger.error(line)
            else:
                self.logger.info(line)

    def data(self, field=None, value=None, indent=2, level='info'):
        """Data line logger

        Parameters
        ----------
        field : str
            Data field name
        value : str
            Data value
        indent : int
            Amount of indention used for the line
        level : str
            Logging level, one of [info,debug,warning,warn,error]
            Default value "info"
        Returns
        -------

        nothing

        """

        if field and value:
            self.line(' ' * indent + '{field:<20} : {value}'.format(field=str(field), value=str(value)), level=level)
        elif field and not value:
            self.line(' ' * indent + '{field:<20}'.format(field=str(field)), level=level)
        elif not field and value:
            self.line(' ' * indent + '{field:<20} : {value}'.format(field=' '*20, value=str(value)), level=level)
        else:
            self.line(' ' * indent, level=level)

    def info(self, text=''):
        """Info line logger

        Parameters
        ----------
        text : str
            Text

        Returns
        -------

        nothing

        """

        self.line(text=text, level='info')

    def debug(self, text=''):
        """Debug line logger

        Parameters
        ----------
        text : str
            Text

        Returns
        -------

        nothing

        """

        self.line(text=text, level='debug')

    def error(self, text=''):
        """Error line logger

        Parameters
        ----------
        text : str
            Text

        Returns
        -------

        nothing

        """

        self.line(text=text, level='error')
