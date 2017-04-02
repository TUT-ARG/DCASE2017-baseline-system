#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils
=====

Utility functions and classes.

Utility functions
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    setup_logging
    get_parameter_hash
    get_class_inheritors
    get_byte_string
    argument_file_exists
    filelist_exists
    posix_path

Timer
^^^^^

.. autosummary::
    :toctree: generated/

    Timer
    Timer.start
    Timer.stop
    Timer.elapsed
    Timer.get_string

SuppressStdoutAndStderr
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    SuppressStdoutAndStderr

"""

import os
import datetime
import time
import hashlib
import json
import locale
import argparse
import logging
import logging.config
import yaml
import pkg_resources


def get_parameter_hash(params):
    """Get unique hash string (md5) for given parameter dict

    Parameters
    ----------
    params : dict, list
        Input parameters

    Returns
    -------
    str
        Unique hash for parameter dict

    """

    md5 = hashlib.md5()
    md5.update(str(json.dumps(params, sort_keys=True)).encode('utf-8'))
    return md5.hexdigest()


def get_class_inheritors(klass):
    """Get all classes inherited from given class

    Parameters
    ----------
    klass : class

    Returns
    -------
    list
        List of classes
    """

    sub_classes = []
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in sub_classes:
                sub_classes.append(child)
                work.append(child)

    return sub_classes


def get_byte_string(num_bytes):
    """Output number of bytes according to locale and with IEC binary prefixes

    Parameters
    ----------
    num_bytes : int > 0 [scalar]
        Bytes

    Returns
    -------
    str
        Human readable string
    """

    KiB = 1024
    MiB = KiB * KiB
    GiB = KiB * MiB
    TiB = KiB * GiB
    PiB = KiB * TiB
    EiB = KiB * PiB
    ZiB = KiB * EiB
    YiB = KiB * ZiB
    locale.setlocale(locale.LC_ALL, '')
    output = locale.format("%d", num_bytes, grouping=True) + ' bytes'

    if num_bytes > YiB:
        output += ' (%.4g YiB)' % (num_bytes / YiB)
    elif num_bytes > ZiB:
        output += ' (%.4g ZiB)' % (num_bytes / ZiB)
    elif num_bytes > EiB:
        output += ' (%.4g EiB)' % (num_bytes / EiB)
    elif num_bytes > PiB:
        output += ' (%.4g PiB)' % (num_bytes / PiB)
    elif num_bytes > TiB:
        output += ' (%.4g TiB)' % (num_bytes / TiB)
    elif num_bytes > GiB:
        output += ' (%.4g GiB)' % (num_bytes / GiB)
    elif num_bytes > MiB:
        output += ' (%.4g MiB)' % (num_bytes / MiB)
    elif num_bytes > KiB:
        output += ' (%.4g KiB)' % (num_bytes / KiB)
    return output


def argument_file_exists(filename):
    """Argument file checker

    Type for argparse. Checks that file exists but does not open.

    Parameters
    ----------
    filename : str

    Returns
    -------
    str
        filename
    """

    if not os.path.exists(filename):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(filename))
    return filename


def setup_logging(parameter_container=None,
                  default_setup_file='logging.yaml',
                  default_level=logging.INFO,
                  environmental_variable='LOG_CFG'):
    """Setup logging configuration

    Parameters
    ----------
    parameter_container : ParameterContainer
        Parameters
    environmental_variable : str
        Environmental variable to get the logging setup filename, if set will override default_setup_file
        Default value "LOG_CFG"
    default_setup_file : str
        Default logging parameter file, used if one is not set in given ParameterContainer
        Default value "logging.yaml"
    default_level : logging.level
        Default logging level, used if one is not set in given ParameterContainer
        Default value "logging.INFO"

    Returns
    -------

    nothing

    """

    if not parameter_container:
        logging_parameter_file = default_setup_file

        value = os.getenv(environmental_variable, None)
        if value:
            # If environmental variable set
            logging_parameter_file = value

        if os.path.exists(logging_parameter_file):
            with open(logging_parameter_file, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

            try:
                import coloredlogs
                coloredlogs.install(level=config['handlers']['console']['level'],
                                    fmt=config['formatters'][config['handlers']['console']['formatter']]['format'],
                                    )
            except ImportError:
                pass
        else:
            logging.basicConfig(level=default_level)
    else:
        logging.config.dictConfig(parameter_container.get('parameters'))
        if parameter_container.get('colored', False) and 'console' in parameter_container.get_path('parameters.handlers'):
            try:
                import coloredlogs
                coloredlogs.install(
                    level=parameter_container.get_path('parameters.handlers.console.level'),
                    fmt=parameter_container.get_path('parameters.formatters')[parameter_container.get_path('parameters.handlers.console.formatter')].get('format')
                )
            except ImportError:
                pass


def filelist_exists(filelist):
    """Check that all file in the list exists

    Parameters
    ----------
    filelist : dict of paths
        List containing paths to files

    Returns
    -------
    bool
        Returns True if all files exists, False if any of them does not
    """

    return all({k: os.path.isfile(v) for k, v in filelist.items()}.values())


def posix_path(path):
    """Converts path to POSIX format

    Parameters
    ----------
    path : str
        Path

    Returns
    -------
    str

    """

    return os.path.normpath(path).replace('\\','/')

def check_pkg_resources(package_requirement, logger=None):
    working_set = pkg_resources.WorkingSet()
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        working_set.require(package_requirement)
    except pkg_resources.VersionConflict:
        message = '{name}: Version conflict, update package [pip install {package_requirement}]'.format(
            name=__name__,
            package_requirement=package_requirement
        )
        logger.exception(message)
        raise
    except pkg_resources.DistributionNotFound:
        message = '{name}: Package not found, install package [pip install {package_requirement}]'.format(
            name=__name__,
            package_requirement=package_requirement
        )
        logger.exception(message)
        raise


class Timer(object):
    """Timer class"""

    def __init__(self):
        # Initialize internal properties
        self._start = None
        self._elapsed = None

    def start(self):
        """Start timer

        Returns
        -------
        self
        """

        self._start = time.time()
        return self

    def stop(self):
        """Stop timer

        Returns
        -------
        self
        """

        self._elapsed = (time.time() - self._start)
        return self

    def elapsed(self):
        """Return elapsed time in seconds since timer was started

        Can be used without stopping the timer

        Returns
        -------
        float
            Seconds since timer was started
        """

        return time.time() - self._start

    def get_string(self):
        """Get elapsed time in a string format

        Returns
        -------
        str
            Time delta between start and stop
        """

        return str(datetime.timedelta(seconds=self._elapsed))

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()


class SuppressStdoutAndStderr(object):
    """Context manager to suppress STDOUT and STDERR

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    After:
    http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        """Assign the null pointers to stdout and stderr.
        """

        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        """Re-assign the real stdout/stderr back
        """

        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
