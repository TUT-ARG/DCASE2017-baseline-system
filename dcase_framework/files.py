#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Files
==================
Utility classes for handling different type of files.

AudioFile
^^^^^^^^^

File class to read audio files. Currently supports wav and flac formats.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    data, fs = AudioFile(filename='test.wav', fs=22050).load()
    # Example 2
    data, fs = AudioFile().load(filename='test.wav', fs=44100, mono=False)

.. autosummary::
    :toctree: generated/

    AudioFile
    AudioFile.load
    AudioFile.save
    AudioFile.exists
    AudioFile.empty

ParameterFile
^^^^^^^^^^^^^

File class to read and write dict based parameter files in YAML format.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    params = ParameterFile(filename='parameters.yaml')
    params.load()
    params.save()
    # Example 2
    params = ParameterFile(filename='parameters.yaml').load()
    params.save()
    # Example 3
    params = ParameterFile({'test':'value'}).save(filename='parameters.yaml')

.. autosummary::
    :toctree: generated/

    ParameterFile
    ParameterFile.load
    ParameterFile.save
    ParameterFile.exists
    ParameterFile.empty


FeatureFile
^^^^^^^^^^^

File class to read and write dict based feature files in cpickle format.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    feat = FeatureFile(filename='features.cpickle')
    feat.load()
    feat.save()
    # Example 2
    feat = FeatureFile(filename='features.cpickle').load()
    feat.save()
    # Example 3
    feat = FeatureFile({'feature':[1,2,3,4]}).save(filename='features.cpickle')

.. autosummary::
    :toctree: generated/

    FeatureFile
    FeatureFile.load
    FeatureFile.save
    FeatureFile.exists
    FeatureFile.empty

DataFile
^^^^^^^^

Generic file class to read and write dict based data files in cpickle format.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    data = DataFile(filename='data.cpickle')
    data.load()
    data.save()
    # Example 2
    data = DataFile(filename='data.cpickle').load()
    data.save()
    # Example 3
    data = DataFile({'data':[1,2,3,4]}).save(filename='data.cpickle')


.. autosummary::
    :toctree: generated/

    DataFile
    DataFile.load
    DataFile.save
    DataFile.exists
    DataFile.empty



RepositoryFile
^^^^^^^^^^^^^^

File class to read and write dict based repositories in cpickle format.

.. autosummary::
    :toctree: generated/

    RepositoryFile
    RepositoryFile.load
    RepositoryFile.exists
    RepositoryFile.empty

TextFile
^^^^^^^^

File class to read and write text files, rows in the text file is stored as items in a list.

.. autosummary::
    :toctree: generated/

    TextFile
    TextFile.load
    TextFile.save
    TextFile.exists
    TextFile.empty

DictFile
^^^^^^^^

Base class for all dict based file classes.

.. autosummary::
    :toctree: generated/

    DictFile
    DictFile.load
    DictFile.save
    DictFile.exists
    DictFile.empty

ListFile
^^^^^^^^

Base class for all list based file classes.

.. autosummary::
    :toctree: generated/

    ListFile
    ListFile.load
    ListFile.save
    ListFile.exists
    ListFile.empty

Mixins
^^^^^^

.. autosummary::
    :toctree: generated/

    FileMixin

"""

from __future__ import print_function, absolute_import
from six import iteritems

import os
import numpy
import logging
import soundfile
import copy
from .decorators import before_and_after_function_wrapper
from .containers import DottedDict, ContainerMixin


class FileMixin(object):
    """Generic file mixin"""
    def get_file_information(self):
        """Get file information, filename

        Returns
        -------
        str

        """

        if self.filename:
            return 'Filename: ['+self.filename+']'
        else:
            return ''

    def detect_file_format(self, filename):
        """Detect file format from extension

        Parameters
        ----------
        filename : str
            filename

        Returns
        -------
        str
            format tag

        Raises
        ------
        IOError:
            Unknown file format

        """

        extension = os.path.splitext(filename.lower())[1]

        file_format = None
        if extension == '.yaml':
            file_format = 'yaml'
        elif extension == '.xml':
            file_format = 'xml'
        elif extension == '.json':
            file_format = 'json'
        elif extension == '.cpickle':
            file_format = 'cpickle'
        elif extension == '.pickle':
            file_format = 'cpickle'
        elif extension == '.pkl':
            file_format = 'cpickle'
        elif extension == '.marshal':
            file_format = 'marshal'
        elif extension == '.msgpack':
            file_format = 'msgpack'
        elif extension == '.txt':
            file_format = 'txt'
        elif extension == '.hash':
            file_format = 'txt'
        elif extension == '.csv':
            file_format = 'csv'
        elif extension == '.ann':
            file_format = 'ann'
        elif extension == '.wav':
            file_format = 'wav'
        elif extension == '.flac':
            file_format = 'flac'
        elif extension == '.mp3':
            file_format = 'mp3'
        elif extension == '.m4a':
            file_format = 'm4a'
        elif extension == '.webm':
            file_format = 'webm'

        if file_format in self.valid_formats:
            return file_format
        else:
            message = '{name}: Unknown format [{format}] for file [{file}]'.format(
                name=self.__class__.__name__,
                format = os.path.splitext(filename)[-1],
                file=filename
            )
            if self.logger:
                self.logger.exception(message)

            raise IOError(message)

    def exists(self):
        """Checks that file exists

        Returns
        -------
        bool

        """

        return os.path.isfile(self.filename)

    def empty(self):
        """Check if file is empty

        Returns
        -------
        bool

        """

        if len(self) == 0:
            return True
        else:
            return False


class DictFile(dict, FileMixin, ContainerMixin):
    """File class inherited from dict, valid file formats [yaml, json, cpickle, marshal, msgpack, txt]"""
    valid_formats = ['yaml', 'json', 'cpickle', 'marshal', 'msgpack', 'txt']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        filename : str, optional
            File path
        logger : logger
            Logger class instance, If none given logger instance will be created
            Default value "None"
        """

        self.filename = kwargs.get('filename', None)
        if self.filename:
            self.format = self.detect_file_format(self.filename)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        if not self.logger.handlers:
            logging.basicConfig()

        dict.__init__(self, *args)

    @before_and_after_function_wrapper
    def load(self, filename=None):
        """Load file

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor

        Raises
        ------
        ImportError:
            Error if file format specific module cannot be imported
        IOError:
            File does not exists or has unknown file format

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        dict.clear(self)
        if self.exists():

            if self.format == 'yaml':
                try:
                    import yaml
                except ImportError:
                    message = '{name}: Unable to import YAML module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                try:
                    with open(self.filename, 'r') as infile:
                        dict.update(self, yaml.load(infile))

                except yaml.YAMLError as exc:
                    self.logger.error("Error while parsing YAML file [%s]" % self.filename)
                    if hasattr(exc, 'problem_mark'):
                        if exc.context is not None:
                            self.logger.error(str(exc.problem_mark) + '\n  ' + str(exc.problem) + ' ' + str(exc.context))
                            self.logger.error('  Please correct data and retry.')
                        else:
                            self.logger.error(str(exc.problem_mark) + '\n  ' + str(exc.problem))
                            self.logger.error('  Please correct data and retry.')
                    else:
                        self.logger.error("Something went wrong while parsing yaml file [%s]" % self.filename)
                    return

            elif self.format == 'cpickle':
                try:
                    import cPickle as pickle
                except ImportError:
                    try:
                        import pickle
                    except ImportError:
                        message = '{name}: Unable to import pickle module.'.format(name=self.__class__.__name__)
                        self.logger.exception(message)
                        raise ImportError(message)

                dict.update(self, pickle.load(open(self.filename, "rb")))

            elif self.format == 'marshal':
                try:
                    import marshal
                except ImportError:
                    message = '{name}: Unable to import marshal module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                dict.update(self, marshal.load(open(self.filename, "rb")))

            elif self.format == 'msgpack':
                try:
                    import msgpack
                except ImportError:
                    message = '{name}: Unable to import msgpack module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                dict.update(self, msgpack.load(open(self.filename, "rb")))

            elif self.format == 'json':
                try:
                    import ujson as json
                except ImportError:
                    try:
                        import json
                    except ImportError:
                        message = '{name}: Unable to import json module.'.format(name=self.__class__.__name__)
                        self.logger.exception(message)
                        raise ImportError(message)

                dict.update(self, json.load(open(self.filename, "r")))

            elif self.format == 'txt':
                with open(self.filename, 'r') as f:
                    lines = f.readlines()
                    dict.update(self, dict(zip(range(0, len(lines)), lines)))

            else:
                message = '{name}: Unknown format [{format}]'.format(name=self.__class__.__name__, format=self.filename)
                self.logger.exception(message)
                raise IOError(message)
        else:
            message = '{name}: File does not exists [{file}]'.format(name=self.__class__.__name__, file=self.filename)
            self.logger.exception(message)
            raise IOError(message)

        return self

    @before_and_after_function_wrapper
    def save(self, filename=None):
        """Save file

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor

        Raises
        ------
        ImportError:
            Error if file format specific module cannot be imported
        IOError:
            File has unknown file format

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        try:
            if hasattr(self, '__getstate__'):
                data = dict(self.__getstate__())
            else:
                data = dict(self)

            if self.format == 'yaml':
                try:
                    import yaml
                except ImportError:
                    message = '{name}: Unable to import yaml module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                with open(self.filename, 'w') as outfile:

                    outfile.write(yaml.dump(self.get_dump_content(data=data), default_flow_style=False))

            elif self.format == 'cpickle':
                try:
                    import cPickle as pickle
                except ImportError:
                    try:
                        import pickle
                    except ImportError:
                        message = '{name}: Unable to import pickle module.'.format(name=self.__class__.__name__)
                        self.logger.exception(message)
                        raise ImportError(message)

                pickle.dump(data, open(self.filename, 'wb'), protocol=2)  # pickle.HIGHEST_PROTOCOL)

            elif self.format == 'marshal':
                try:
                    import marshal
                except ImportError:
                    message = '{name}: Unable to import marshal module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                marshal.dump(data, open(self.filename, 'wb'))

            elif self.format == 'msgpack':
                try:
                    import msgpack
                except ImportError:
                    message = '{name}: Unable to import msgpack module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                msgpack.dump(data, open(self.filename, 'wb'), use_bin_type=True)

            elif self.format == 'json':
                try:
                    import ujson as json
                except ImportError:
                    try:
                        import json
                    except ImportError:
                        message = '{name}: Unable to import json module.'.format(name=self.__class__.__name__)
                        self.logger.exception(message)
                        raise ImportError(message)

                json.dump(data, open(self.filename, 'wb'))

            elif self.format == 'txt':
                with open(self.filename, "w") as text_file:
                    for line_id in self:
                        text_file.write(self[line_id])
            else:
                message = '{name}: Unknown format [{format}]'.format(name=self.__class__.__name__, format=self.filename)
                self.logger.exception(message)
                raise IOError(message)

        except KeyboardInterrupt:
            os.remove(self.filename)        # Delete the file, since most likely it was not saved fully
            raise

    def get_dump_content(self, data):
        """Clean internal content for saving

        Numpy, DottedDict content is converted to standard types

        Parameters
        ----------
        data : dict

        Returns
        -------

        dict

        """
        if data:
            data = dict(data)
            for k, v in iteritems(data):
                if isinstance(v, numpy.generic):
                    data[k] = numpy.asscalar(v)
                elif isinstance(v, DottedDict):
                    data[k] = self.get_dump_content(data=dict(data[k]))
                elif isinstance(v, dict):
                    data[k] = self.get_dump_content(data=data[k])

            return data


class ListFile(list, FileMixin):
    """File class inherited from list, valid file formats [txt]"""
    valid_formats = ['txt', 'yaml']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        filename : str, optional
            File path
        logger : logger
            Logger class instance, If none given logger instance will be created
            Default value "None"
        """

        self.filename = kwargs.get('filename', None)
        if self.filename:
            self.format = self.detect_file_format(self.filename)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        if not self.logger.handlers:
            logging.basicConfig()

        list.__init__(self, *args)

    @before_and_after_function_wrapper
    def load(self, filename=None):
        """Load file

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor

        Raises
        ------
        IOError:
            File does not exists or has unknown file format

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        if self.exists():
            if self.format == 'txt':
                with open(self.filename, 'r') as f:
                    lines = f.readlines()
                    # Remove line breaks
                    for i in range(0, len(lines)):
                        lines[i] = lines[i].replace('\n', '')
                    list.__init__(self, lines)

            elif self.format == 'yaml':
                try:
                    import yaml
                except ImportError:
                    message = '{name}: Unable to import YAML module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                try:
                    with open(self.filename, 'r') as infile:
                        data = yaml.load(infile)
                        if isinstance(data, list):
                            list.__init__(self, data)
                        else:
                            message = '{name}: YAML data is not in list format.'.format(name=self.__class__.__name__)
                            self.logger.exception(message)
                            raise ImportError(message)

                except yaml.YAMLError as exc:
                    self.logger.error("Error while parsing YAML file [%s]" % self.filename)
                    if hasattr(exc, 'problem_mark'):
                        if exc.context is not None:
                            self.logger.error(str(exc.problem_mark) + '\n  ' + str(exc.problem) + ' ' + str(exc.context))
                            self.logger.error('  Please correct data and retry.')
                        else:
                            self.logger.error(str(exc.problem_mark) + '\n  ' + str(exc.problem))
                            self.logger.error('  Please correct data and retry.')
                    else:
                        self.logger.error("Something went wrong while parsing yaml file [%s]" % self.filename)
                    return

            else:
                message = '{name}: Unknown format [{format}]'.format(name=self.__class__.__name__, format=self.filename)
                self.logger.exception(message)
                raise IOError(message)
        else:
            message = '{name}: File does not exists [{file}]'.format(name=self.__class__.__name__, file=self.filename)
            self.logger.exception(message)
            raise IOError(message)

        return self

    @before_and_after_function_wrapper
    def save(self, filename=None):
        """Save file

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor

        Raises
        ------
        IOError:
            File has unknown file format

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)
        try:
            if self.format == 'yaml':
                try:
                    import yaml
                except ImportError:
                    message = '{name}: Unable to import yaml module.'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ImportError(message)

                with open(self.filename, 'w') as outfile:
                    data = copy.deepcopy(list(self))
                    for item_id, item in enumerate(data):
                        data[item_id] = self.get_dump_content(data=item)

                    outfile.write(yaml.dump(data, default_flow_style=False))

            elif self.format == 'txt':
                with open(self.filename, "w") as text_file:
                    for line in self:
                        text_file.write(line+'\n')
            else:
                message = '{name}: Unknown format [{format}]'.format(name=self.__class__.__name__, format=self.filename)
                self.logger.exception(message)
                raise IOError(message)

        except KeyboardInterrupt:
            os.remove(self.filename)            # Delete the file, since most likely it was not saved fully
            raise

    def get_dump_content(self, data):
        """Clean internal content for saving

        Numpy, DottedDict content is converted to standard types

        Parameters
        ----------
        data : dict

        Returns
        -------

        dict

        """
        if data:
            data = dict(data)
            for k, v in iteritems(data):
                if isinstance(v, numpy.generic):
                    data[k] = numpy.asscalar(v)
                elif isinstance(v, DottedDict):
                    data[k] = self.get_dump_content(data=dict(data[k]))
                elif isinstance(v, dict):
                    data[k] = self.get_dump_content(data=data[k])

            return data


class AudioFile(FileMixin):
    """File class for audio files, valid file formats  [wav, flac]"""
    valid_formats = ['wav', 'flac', 'm4a', 'webm']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        fs : int
            Target sampling frequency, if loaded audio does have different sampling frequency, audio will be re-sampled.
            Default value "44100"
        mono : bool
            Monophonic target, multi-channel audio will be down-mixed.
            Default value "True"
        filename : str, optional
            File path
        logger : logger
            Logger class instance, If none given logger instance will be created
            Default value "None"
        """

        self.data = kwargs.get('data', None)                     # Audio data itself

        self.filename = kwargs.get('filename', None)
        if self.filename:
            self.format = self.detect_file_format(self.filename)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        if not self.logger.handlers:
            logging.basicConfig()

        self.fs = kwargs.get('fs', 44100)
        self.mono = kwargs.get('mono', True)

    @before_and_after_function_wrapper
    def load(self, filename=None, fs=None, mono=None, res_type='kaiser_best', start=None, stop=None):
        """Load file

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor
        fs : int
            Target sampling frequency, if loaded audio does have different sampling frequency, audio will be re-sampled.
            Default value one given to class constructor
        mono : bool
            Monophonic target, multi-channel audio will be down-mixed.
            Default value one given to class constructor
        res_type : str
            Resample type, defined by Librosa
            Default value "kaiser_best"
        start : float, optional
            Segment start time in seconds
            Default value "None"
        stop : float, optional
            Segment stop time in seconds
            Default value "None"

        Raises
        ------
        IOError:
            File does not exists or has unknown file format

        Returns
        -------
        self

        """

        if filename is not None:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        if self.exists():
            if fs is not None:
                self.fs = fs

            if mono is not None:
                self.mono = mono

            if self.format == 'wav':
                info = soundfile.info(file=self.filename)

                # Handle segment start and stop
                if start is not None and stop is not None:
                    start_sample = int(start * info.samplerate)
                    stop_sample = int(stop * info.samplerate)
                    if stop_sample > info.frames:
                        stop_sample = info.frames
                else:
                    start_sample = None
                    stop_sample = None

                self.data, source_fs = soundfile.read(file=self.filename, start=start_sample, stop=stop_sample)
                self.data = self.data.T

                # Down-mix audio
                if self.mono and len(self.data.shape) > 1:
                    self.data = numpy.mean(self.data, axis=0)

                # Resample
                if self.fs != source_fs:
                    import librosa
                    self.data = librosa.core.resample(self.data, source_fs, self.fs, res_type=res_type)

            elif self.format in ['flac', 'm4a', 'webm']:
                import librosa
                if start is not None and stop is not None:
                    offset = start
                    duration = stop - start
                else:
                    offset = 0.0
                    duration = None
                self.data, self.fs = librosa.load(self.filename, sr=self.fs, mono=self.mono, res_type=res_type, offset=offset, duration=duration)

            else:
                message = '{name}: Unknown format [{format}]'.format(name=self.__class__.__name__, format=self.filename)
                self.logger.exception(message)
                raise IOError(message)
        else:
            message = '{name}: File does not exists [{file}]'.format(name=self.__class__.__name__, file=self.filename)
            self.logger.exception(message)
            raise IOError(message)

        return self.data, self.fs

    def save(self, filename=None, bitdepth=16):
        """Save audio

        Parameters
        ----------
        filename : str, optional
            File path
            Default value filename given to class constructor
        bitdepth : int, optional
            Bit depth for audio
            Default value "16"
        Raises
        ------
        ImportError:
            Error if file format specific module cannot be imported
        IOError:
            File has unknown file format

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        if self.format == 'wav':
            if bitdepth == 16:
                soundfile.write(file=self.filename,
                                data=self.data,
                                samplerate=self.fs,
                                subtype='PCM_16')

            elif bitdepth == 24:
                soundfile.write(file=self.filename,
                                data=self.data,
                                samplerate=self.fs,
                                subtype='PCM_24')

            elif bitdepth == 32:
                soundfile.write(file=self.filename,
                                data=self.data,
                                samplerate=self.fs,
                                subtype='PCM_32')

            elif bitdepth is None:
                soundfile.write(file=self.filename,
                                data=self.data,
                                samplerate=self.fs)

            else:
                message = '{name}: Unexpected bit depth [{bitdepth}]'.format(name=self.__class__.__name__,
                                                                             bitdepth=bitdepth)
                self.logger.exception(message)
                raise IOError(message)

        elif self.format == 'flac':
            soundfile.write(file=self.filename,
                            data=self.data,
                            samplerate=self.fs)

        else:
            message = '{name}: Unknown format for saving [{format}]'.format(name=self.__class__.__name__,
                                                                            format=self.filename)
            self.logger.exception(message)
            raise IOError(message)


class TextFile(ListFile):
    """File class for text files, Inherited from ListFile, valid file formats [txt]"""
    valid_formats = ['txt']


class DataFile(DictFile):
    """File class for data files, Inherited from DictFile, valid file formats [cpickle]"""
    valid_formats = ['cpickle']


class ParameterFile(DictFile):
    """File class for parameter files, Inherited from DictFile, valid file formats [yaml]"""
    valid_formats = ['yaml']


class ParameterListFile(ListFile):
    """File class for parameter list files, Inherited from ListFile, valid file formats [yaml]"""
    valid_formats = ['yaml']


class FeatureFile(DictFile):
    """File class for feature files, Inherited from DictFile, valid file formats [cpickle]"""
    valid_formats = ['cpickle']


class RepositoryFile(DictFile):
    """File class for repository files, Inherited from DictFile, valid file formats [cpickle]"""
    valid_formats = ['cpickle']


