#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features
==================
Classes for feature handling

FeatureContainer
^^^^^^^^^^^^^^^^

Container class to store features along with statistics and meta data. Class is based on dict through
inheritance of FeatureFile class.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    feature_container = FeatureContainer(filename='features.cpickle')
    feature_container.show()
    feature_container.log()
    print('Feature shape={shape}'.format(shape=feature_container.shape))
    print('Feature channels={channels}'.format(channels=feature_container.channels))
    print('Feature frames={frames}'.format(frames=feature_container.frames))
    print('Feature vector length={vector_length}'.format(vector_length=feature_container.vector_length))
    print(feature_container.feat)
    print(feature_container.stat)
    print(feature_container.meta)
    # Example 2
    feature_container = FeatureContainer().load(filename='features.cpickle')
    # Example 3
    feature_repository = FeatureContainer().load(filename_list={'mel':'mel_features.cpickle', 'mfcc':'mfcc_features.cpickle'})
    # Example 4
    feature_container = FeatureContainer(features=[numpy.ones((100,10)),numpy.ones((100,10))])

.. autosummary::
    :toctree: generated/

    FeatureContainer
    FeatureContainer.show
    FeatureContainer.log
    FeatureContainer.get_path
    FeatureContainer.shape
    FeatureContainer.channels
    FeatureContainer.frames
    FeatureContainer.vector_length
    FeatureContainer.feat
    FeatureContainer.stat
    FeatureContainer.meta
    FeatureContainer.load


FeatureRepository
^^^^^^^^^^^^^^^^^

Feature repository class, where feature containers for each type of features are stored in a dict. Type name is
used as key.

.. autosummary::
    :toctree: generated/

    FeatureRepository
    FeatureRepository.show
    FeatureRepository.log
    FeatureRepository.get_path
    FeatureRepository.load

FeatureExtractor
^^^^^^^^^^^^^^^^

Feature extractor class.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1, to get feature only without storing them
    feature_repository = FeatureExtractor().extract(audio_file='debug/test.wav',
                                                    extractor_name='mfcc',
                                                    extractor_params={
                                                        'mfcc': {
                                                            'n_mfcc': 10
                                                        }
                                                    }
                                                    )
    feature_repository['mfcc'].show()

    # Example 2, to store features during the extraction
    feature_repository = FeatureExtractor(store=True).extract(
        audio_file='debug/test.wav',
        extractor_name='mfcc',
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        },
        storage_paths={
            'mfcc': 'debug/test.mfcc.cpickle'
        }
    )

    # Example 3
    print(FeatureExtractor().get_default_parameters())


.. autosummary::
    :toctree: generated/

    FeatureExtractor
    FeatureExtractor.extract
    FeatureExtractor.get_default_parameters

FeatureNormalizer
^^^^^^^^^^^^^^^^^

Feature normalizer class.

Usage examples:

.. code-block:: python
    :linenos:

    # Example 1
    normalizer = FeatureNormalizer()
    for feature_matrix in training_items:
         normalizer.accumulate(feature_matrix)
    normalizer.finalize()

    for feature_matrix in test_items:
        feature_matrix_normalized = normalizer.normalize(feature_matrix)
        # used the features

    # Example 2
    with FeatureNormalizer() as norm:
        norm.accumulate(feature_repository['mfcc'])

    for feature_matrix in test_items:
        feature_matrix_normalized = normalizer.normalize(feature_matrix)
        # used the features

.. autosummary::
    :toctree: generated/

    FeatureNormalizer
    FeatureNormalizer.accumulate
    FeatureNormalizer.finalize
    FeatureNormalizer.normalize
    FeatureNormalizer.process

FeatureStacker
^^^^^^^^^^^^^^

Feature stacking class. Class takes feature vector recipe and FeatureRepository, and creates appropriate feature matrix.


**Feature vector recipe**

With a recipe one can either select full matrix, only part of with start and end index, or select individual rows from it.

Example recipe:

.. code-block:: python
    :linenos:

    [
     {
        'method': 'mfcc',
     },
     {
        'method': 'mfcc_delta'
        'vector-index: {
            'channel': 0,
            'start': 1,
            'end': 17,
            'full': False,
            'selection': False,
        }
      },
     {
        'method': 'mfcc_acceleration',
        'vector-index: {
            'channel': 0,
            'full': False,
            'selection': True,
            'vector': [2, 4, 6]
        }
     }
    ]

See  :py:meth:`dcase_framework.ParameterContainer._parse_recipe` how text recipe can be confiniently used to generate
above structure.

.. autosummary::
    :toctree: generated/

    FeatureStacker
    FeatureStacker.normalizer
    FeatureStacker.feature_vector
    FeatureStacker.process


FeatureAggregator
^^^^^^^^^^^^^^^^^

Feature aggregator can be used to process feature matrix in a processing windows.
This processing stage can be used to collapse features within certain window lengths by
calculating mean and std of them, or flatten the matrix into one feature vector.

Supported processing methods:

- ``flatten``
- ``mean``
- ``std``
- ``cov``
- ``kurtosis``
- ``skew``

The processing methods can combined.

Usage examples:

.. code-block:: python
    :linenos:

    feature_aggregator = FeatureAggregator(
        recipe=['mean', 'std'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_list={'mfcc': 'mfcc.cpickle'})
    feature_matrix = feature_stacker.feature_vector(feature_repository=feature_repository)
    feature_matrix = feature_aggregator.process(feature_container=feature_matrix)

.. autosummary::
    :toctree: generated/

    FeatureAggregator
    FeatureAggregator.process

FeatureMasker
^^^^^^^^^^^^^

Feature masker can be used to mask segments of feature matrix out. For examples, error segments of signal
can be excluded from the matrix.

Usage examples:

.. code-block:: python
    :linenos:

    feature_masker = FeatureMasker(hop_length_seconds=0.01)
    mask_events = MetaDataContainer([
        {
            'event_onset': 1.0,
            'event_offset': 1.5,
        },
        {
            'event_onset': 2.0,
            'event_offset': 2.5,
        },
    ])

    masked_features = feature_masker.process(feature_repository=feature_repository, mask_events=mask_events)

.. autosummary::
    :toctree: generated/

    FeatureMasker
    FeatureMasker.process

"""

from __future__ import print_function, absolute_import
from six import iteritems
import os
import logging
import numpy
import librosa
import scipy
import collections
import copy
from time import gmtime, strftime
from .files import FeatureFile, AudioFile, DataFile, RepositoryFile
from .containers import ContainerMixin, DottedDict
from .parameters import ParameterContainer
from .utils import filelist_exists
from .metadata import MetaDataContainer


class FeatureContainer(FeatureFile, ContainerMixin):
    """Feature container inherited from dict

    Container has following internal structure:

    - feat, list of feature matrices, [channel][frames,feature_vector]
    - stat, list of feature statistics
    - meta, dict with feature meta data

    """
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        filename: str, optional
            If filename is given container is loaded in the initialization stage.
            Default value "None"

        features: list, optional

        """
        if kwargs.get('filename', None):
            super(FeatureContainer, self).__init__({
                'feat': [],
                'stat': None,
                'meta': {},
            })
            self.load(filename=kwargs.get('filename'))
        else:
            if kwargs.get('features', []):
                super(FeatureContainer, self).__init__({
                    'feat': kwargs.get('features', []),
                    'stat': None,
                    'meta': kwargs.get('meta', {}),
                })
            else:
                super(FeatureContainer, self).__init__(*args, **kwargs)

    @property
    def shape(self):
        """Shape of feature matrix

        Returns
        -------

        """

        if 'feat' in self:
            return self.feat[0].shape
        else:
            return None

    @property
    def channels(self):
        """Number of feature channels

        Returns
        -------
            int

        """

        if 'feat' in self:
            return len(self.feat)
        else:
            return None

    @property
    def frames(self):
        """Number of feature frames

        Returns
        -------
            int

        """

        if 'feat' in self:
            return self.feat[0].shape[0]
        else:
            return None

    @property
    def vector_length(self):
        """Feature vector length

        Returns
        -------
            int

        """

        if 'feat' in self:
            return self.feat[0].shape[1]
        else:
            return None

    @property
    def feat(self):
        """Feature data

        Returns
        -------
            list of numpy.ndarray

        """

        if 'feat' in self:
            return self['feat']
        else:
            return None

    @feat.setter
    def feat(self, value):
        self['feat'] = value

    @property
    def stat(self):
        """Statistics of feature data

        Returns
        -------
            list of dicts
        """

        if self.feat:
            if 'stat' not in self or not self['stat']:
                stat_container = []
                for channel_data in self.feat:
                    stat_container.append({
                        'mean': numpy.mean(channel_data, axis=0),
                        'std': numpy.std(channel_data, axis=0),
                        'N': channel_data.shape[0],
                        'S1': numpy.sum(channel_data, axis=0),
                        'S2': numpy.sum(channel_data ** 2, axis=0),
                    })
                self['stat'] = stat_container
            return self['stat']
        else:
            return None

    @property
    def meta(self):
        """Meta data

        Returns
        -------
            dict
        """
        if 'meta' in self:
            return self['meta']
        else:
            return None

    @meta.setter
    def meta(self, value):
        self['meta'] = value

    def load(self, filename=None, filename_dict=None):
        """Load data into container

        If filename is given, container is loaded from disk
        If filename_list is given, a FeatureRepository is created and returned.

        Parameters
        ----------
        filename : str, optional
        filename_dict : dict, optional

        Returns
        -------
            FeatureContainer or FeatureRepository
        """

        if filename:
            return super(FeatureContainer, self).load(filename=filename)

        if filename_dict:
            repository = FeatureRepository({})
            for method, filename in iteritems(filename_dict):
                repository[method] = FeatureContainer().load(filename=filename)

            return repository


class FeatureRepository(RepositoryFile, ContainerMixin):
    """Feature repository

    Feature containers for each type of features are stored in a dict. Type name is used as key.

    """
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        filename_dict: dict
            Dict of file paths, feature extraction method label as key, and filename as value.
            If given, features are loaded in the initialization stage.
            Default value "None"

        features: list, optional

        """

        super(FeatureRepository, self).__init__(*args, **kwargs)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        if kwargs.get('filename_dict', None):
            self.filename_dict = kwargs.get('filename_dict', None)
            self.load()

    def load(self, filename_dict=None):
        """Load file list

        Parameters
        ----------
        filename_dict : dict
            Dict of file paths, feature extraction method label as key, and filename as value.

        Returns
        -------
        self

        """

        if filename_dict:
            self.filename_dict = filename_dict

        if self.filename_dict and filelist_exists(self.filename_dict):
            dict.clear(self)
            sorted(self.filename_dict)
            for method, filename in iteritems(self.filename_dict):
                if not method.startswith('_'):
                    # Skip method starting with '_', those are just for extra info
                    self[method] = FeatureContainer().load(filename=filename)

            return self

        else:
            message = '{name}: Feature repository cannot be loaded [{filename_dict}]'.format(
                name=self.__class__.__name__,
                filename_dict=self.filename_dict
            )
            self.logger.exception(message)
            raise IOError(message)


class FeatureExtractor(object):
    """Feature extractor"""
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        store : bool
            Store features to disk
            Default value "False"
        overwrite : bool
            If set True, features are overwritten on disk
            Default value "False"

        """

        self.eps = numpy.spacing(1)
        self.overwrite = kwargs.get('overwrite', False)
        self.store = kwargs.get('store', False)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        self.valid_extractors = [
            'mfcc',
            'mfcc_delta',
            'mfcc_acceleration',
            'mel'
        ]
        self.valid_extractors += kwargs.get('valid_extractors', [])

        self.default_general_parameters = {
            'fs': 44100,
            'win_length_samples': int(0.04 * 44100),
            'hop_length_samples': int(0.02 * 44100),
        }
        self.default_general_parameters.update(kwargs.get('default_general_parameters', {}))

        self.default_parameters = {
            'mfcc': {
                'mono': True,  # [True, False]
                'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
                'spectrogram_type': 'magnitude',  # [magnitude, power]
                'n_mfcc': 20,  # Number of MFCC coefficients
                'n_mels': 40,  # Number of MEL bands used
                'n_fft': 2048,  # FFT length
                'fmin': 0,  # Minimum frequency when constructing MEL bands
                'fmax': 22050,  # Maximum frequency when constructing MEL band
                'htk': False,  # Switch for HTK-styled MEL-frequency equation
            },
            'mfcc_delta': {
                'width': 9,
                'dependency_method': 'mfcc',
            },
            'mfcc_acceleration': {
                'width': 9,
                'dependency_method': 'mfcc',
            },
            'mel': {
                'mono': True,  # [True, False]
                'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
                'spectrogram_type': 'magnitude',  # [magnitude, power]
                'n_mels': 40,  # Number of MEL bands used
                'normalize_mel_bands': False,  # [True, False]
                'n_fft': 2048,  # FFT length
                'fmin': 0,  # Minimum frequency when constructing MEL bands
                'fmax': 22050,  # Maximum frequency when constructing MEL band
                'htk': True,  # Switch for HTK-styled MEL-frequency equation
                'log': True,  # Logarithmic
            }
        }
        self.default_parameters.update(kwargs.get('default_parameters', {}))

        # Update general parameters and expand dependencies
        for method, data in iteritems(self.default_parameters):
            data.update(self.default_general_parameters)
            if ('dependency_method' in data and
               data['dependency_method'] in self.valid_extractors and
               data['dependency_method'] in self.default_parameters):

                data['dependency_parameters'] = self.default_parameters[data['dependency_method']]

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'eps': self.eps,
            'overwrite': self.overwrite,
            'store': self.store,
            'valid_extractors': self.valid_extractors,
            'default_general_parameters': self.default_general_parameters,
            'default_parameters': self.default_parameters,
        }

    def __setstate__(self, d):
        self.eps = d['eps']
        self.overwrite = d['overwrite']
        self.store = d['store']
        self.valid_extractors = d['valid_extractors']
        self.default_general_parameters = d['default_general_parameters']
        self.default_parameters = d['default_parameters']
        self.logger = logging.getLogger(__name__)

    def extract(self, audio_file, extractor_params=None, storage_paths=None, extractor_name=None):
        """Extract features for audio file

        Parameters
        ----------
        audio_file : str
            Filename of audio file.
        extractor_params : dict of dicts
            Keys at first level corresponds to feature extraction methods, and second level is parameters given to the
            extractor method. If none given, default parameters used.
        storage_paths : dict of strings
            Keys at first level corresponds to feature extraction methods, second level is path to store feature
            containers.
        extractor_name : str
            Feature extractor method name, if none given, extractor_params is used. Use this to select specific
            extractor method.
            Default value "None"

        Raises
        ------
        ValueError:
            Unknown extractor method

        Returns
        -------
        FeatureRepository
            Repository, a dict of FeatureContainers

        """

        if extractor_params is None:
            extractor_params = {}

        if storage_paths is None:
            storage_paths = {}

        # Get extractor list
        if extractor_name is None:
            extractor_list = list(extractor_params.keys())
        else:
            extractor_list = [extractor_name]

            if extractor_name in extractor_params:
                extractor_params = {
                    extractor_name: extractor_params[extractor_name]
                }

        # Update (recursively) internal default parameters with given parameters
        extractor_params = self._update(self.default_parameters, extractor_params)

        # Update general parameters and expand dependencies
        for method, data in iteritems(extractor_params):
            if ('dependency_method' in data and
               data['dependency_method'] in self.valid_extractors and
               data['dependency_method'] in extractor_params):

                data['dependency_parameters'] = extractor_params[data['dependency_method']]

        feature_repository = FeatureRepository({})
        for extractor_name in extractor_list:
            if extractor_name not in self.valid_extractors:
                message = '{name}: Invalid extractor method [{method}]'.format(
                    name=self.__class__.__name__,
                    method=extractor_name
                )

                self.logger.exception(message)
                raise ValueError(message)

            current_extractor_params = extractor_params[extractor_name]

            extract = True
            # Check do we need to extract anything
            if not self.overwrite and extractor_name in storage_paths and os.path.isfile(storage_paths[extractor_name]):
                # Load from disk
                feature_repository[extractor_name] = FeatureContainer(filename=storage_paths[extractor_name])

                # Check the parameters
                hash1 = ParameterContainer().get_hash(current_extractor_params)
                hash2 = ParameterContainer().get_hash(feature_repository[extractor_name]['meta']['parameters'])
                if hash1 == hash2:
                    # The loaded data contains features with same parameters, no need to extract them anymore
                    extract = False

            # Feature extraction stage
            if extract:
                # Load audio
                y, fs = self._load_audio(audio_file=audio_file, params=current_extractor_params)

                # Check for dependency to other features
                if 'dependency_method' in current_extractor_params and current_extractor_params['dependency_method']:
                    # Current extractor is depending on other extractor

                    if current_extractor_params['dependency_method'] not in self.valid_extractors:
                        message = '{name}: Invalid dependency extractor method [{method1}] for method [{method2}]'.format(
                            name=self.__class__.__name__,
                            method1=current_extractor_params['dependency_method'],
                            method2=extractor_name
                        )

                        self.logger.exception(message)
                        raise ValueError(message)

                    if (current_extractor_params['dependency_method'] in storage_paths and
                       os.path.isfile(storage_paths[current_extractor_params['dependency_method']])):

                        # Load features from disk
                        data = FeatureContainer(
                            filename=storage_paths[current_extractor_params['dependency_method']]
                        ).feat

                    else:
                        # Extract features
                        dependency_func = getattr(self, '_{}'.format(current_extractor_params['dependency_method']), None)
                        if dependency_func is not None:
                            data = dependency_func(data=y, params=current_extractor_params['dependency_parameters'])
                        else:
                            message = '{name}: No extraction method for dependency extractor [{method}]'.format(
                                name=self.__class__.__name__,
                                method=current_extractor_params['dependency_method']
                            )

                            self.logger.exception(message)
                            raise ValueError(message)

                else:
                    # By pass
                    data = y

                # Extract features
                extractor_func = getattr(self, '_{}'.format(extractor_name), None)
                if extractor_func is not None:
                    data = extractor_func(data=data, params=current_extractor_params)

                    # Feature extraction meta information
                    meta = {
                        'parameters': current_extractor_params,
                        'datetime': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                        'audio_file': audio_file,
                        'extractor_version': self.__version__,
                    }

                    # Create feature container
                    feature_container = FeatureContainer(features=data, meta=meta)
                    if self.store and extractor_name in storage_paths:
                        feature_container.save(filename=storage_paths[extractor_name])
                    feature_repository[extractor_name] = feature_container
                else:
                    message = '{name}: No extraction method for extractor [{method}]'.format(
                        name=self.__class__.__name__,
                        method=extractor_name
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

        return FeatureRepository(feature_repository)

    def get_default_parameters(self):
        """Get default parameters as dict

        Returns
        -------
        DottedDict

        """

        return DottedDict(self.default_parameters)

    def _mel(self, data, params):
        """Mel-band energies

        Parameters
        ----------
        data : numpy.ndarray
            Audio data.
        params : dict
            Parameters.

        Returns
        -------
        list of numpy.ndarray
            List of feature matrices, feature matrix per audio channel.

        """

        window = self._window_function(
            N=params.get('win_length_samples'),
            window_type=params.get('window')
        )

        mel_basis = librosa.filters.mel(
            sr=params.get('fs'),
            n_fft=params.get('n_fft'),
            n_mels=params.get('n_mels'),
            fmin=params.get('fmin'),
            fmax=params.get('fmax'),
            htk=params.get('htk')
        )

        if params.get('normalize_mel_bands'):
            mel_basis /= numpy.max(mel_basis, axis=-1)[:, None]

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            spectrogram_ = self._spectrogram(
                y=data[channel, :],
                n_fft=params.get('n_fft'),
                win_length_samples=params.get('win_length_samples'),
                hop_length_samples=params.get('hop_length_samples'),
                spectrogram_type=params.get('spectrogram_type') if 'spectrogram_type' in params else 'magnitude',
                center=True,
                window=window
            )

            mel_spectrum = numpy.dot(mel_basis, spectrogram_)
            if params.get('log'):
                mel_spectrum = numpy.log(mel_spectrum + self.eps)

            mel_spectrum = mel_spectrum.T

            feature_matrix.append(mel_spectrum)

        return feature_matrix

    def _mfcc(self, data, params):
        """Static MFCC

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------
        list of numpy.ndarray
            List of feature matrices, feature matrix per audio channel

        """

        window = self._window_function(
            N=params.get('win_length_samples'),
            window_type=params.get('window')
        )

        mel_basis = librosa.filters.mel(
            sr=params.get('fs'),
            n_fft=params.get('n_fft'),
            n_mels=params.get('n_mels'),
            fmin=params.get('fmin'),
            fmax=params.get('fmax'),
            htk=params.get('htk')
        )

        if params.get('normalize_mel_bands'):
            mel_basis /= numpy.max(mel_basis, axis=-1)[:, None]

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            # Calculate Static Coefficients
            spectrogram_ = self._spectrogram(
                y=data[channel, :],
                n_fft=params.get('n_fft'),
                win_length_samples=params.get('win_length_samples'),
                hop_length_samples=params.get('hop_length_samples'),
                spectrogram_type=params.get('spectrogram_type') if 'spectrogram_type' in params else 'magnitude',
                center=True,
                window=window
            )

            mel_spectrum = numpy.dot(mel_basis, spectrogram_)

            mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                        n_mfcc=params.get('n_mfcc'))

            feature_matrix.append(mfcc.T)

        return feature_matrix

    def _mfcc_delta(self, data, params):
        """Delta MFCC

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------
        list of numpy.ndarray
            List of feature matrices, feature matrix per audio channel

        """

        feature_matrix = []
        for channel in range(0, len(data)):
            # Delta coefficients
            delta = librosa.feature.delta(
                data[channel].T,
                width=params.get('width')
            )

            feature_matrix.append(delta.T)

        return feature_matrix

    def _mfcc_acceleration(self, data, params):
        """Acceleration MFCC

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------
        list of numpy.ndarray
            List of feature matrices, feature matrix per audio channel

        """

        feature_matrix = []
        for channel in range(0, len(data)):
            # Acceleration coefficients (aka delta delta)
            acceleration = librosa.feature.delta(
                data[channel].T,
                order=2,
                width=params.get('width')
            )

            feature_matrix.append(acceleration.T)

        return feature_matrix

    def _load_audio(self, audio_file, params):
        """Load audio using AudioFile class

        Parameters
        ----------
        audio_file : str
        params : dict

        Returns
        -------
        numpy.ndarray
            Audio data

        fs : int
            Sampling frequency

        """

        # Collect parameters
        mono = False
        if 'mono' in params:
            mono = params.get('mono')

        elif 'dependency_parameters' in params and 'mono' in params['dependency_parameters']:
            mono = params['dependency_parameters']['mono']

        fs = None
        if 'fs' in params:
            fs = params.get('fs')

        elif 'dependency_parameters' in params and 'fs' in params['dependency_parameters']:
            fs = params['dependency_parameters']['fs']

        normalize_audio = False
        if 'normalize_audio' in params:
            normalize_audio = params.get('normalize_audio')

        elif 'dependency_parameters' in params and 'normalize_audio' in params['dependency_parameters']:
            normalize_audio = params['dependency_parameters']['normalize_audio']

        # Load audio with correct parameters
        y, fs = AudioFile().load(filename=audio_file, mono=mono, fs=fs)

        if mono:
            # Make sure mono audio has correct shape
            y = numpy.reshape(y, [1, -1])

        # Normalize audio
        if normalize_audio:
            for channel in range(0, y.shape[0]):
                y[channel] = self._normalize_audio(y[channel])

        return y, fs

    @staticmethod
    def _normalize_audio(y, head_room=0.005):
        """Normalize audio

        Parameters
        ----------
        y : numpy.ndarray
            Audio data
        head_room : float
            Head room

        Returns
        -------
        numpy.ndarray
            Audio data

        """

        mean_value = numpy.mean(y)
        y -= mean_value

        max_value = max(abs(y)) + head_room
        return y / max_value

    def _window_function(self, N, window_type='hamming_asymmetric'):
        """Window function

        Parameters
        ----------
        N : int
            window length

        window_type : str
            window type
            (Default value='hamming_asymmetric')
        Raises
        ------
        ValueError:
            Unknown window type

        Returns
        -------
            window function : array
        """

        # Windowing function
        if window_type == 'hamming_asymmetric':
            return scipy.signal.hamming(N, sym=False)
        elif window_type == 'hamming_symmetric':
            return scipy.signal.hamming(N, sym=True)
        elif window_type == 'hann_asymmetric':
            return scipy.signal.hann(N, sym=False)
        elif window_type == 'hann_symmetric':
            return scipy.signal.hann(N, sym=True)
        else:
            message = '{name}: Unknown window type [{window_type}]'.format(
                name=self.__class__.__name__,
                window_type=window_type
            )

            self.logger.exception(message)
            raise ValueError(message)

    def _spectrogram(self, y,
                     n_fft=1024,
                     win_length_samples=0.04,
                     hop_length_samples=0.02,
                     window=scipy.signal.hamming(1024, sym=False),
                     center=True,
                     spectrogram_type='magnitude'):
        """Spectrogram

        Parameters
        ----------
        y : numpy.ndarray
            Audio data
        n_fft : int
            FFT size
            Default value "1024"
        win_length_samples : float
            Window length in seconds
            Default value "0.04"
        hop_length_samples : float
            Hop length in seconds
            Default value "0.02"
        window : array
            Window function
            Default value "scipy.signal.hamming(1024, sym=False)"
        center : bool
            If true, input signal is padded so to the frame is centered at hop length
            Default value "True"
        spectrogram_type : str
            Type of spectrogram "magnitude" or "power"
            Default value "magnitude"

        Returns
        -------
        np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
            STFT matrix

        """

        if spectrogram_type == 'magnitude':
            return numpy.abs(librosa.stft(y + self.eps,
                                          n_fft=n_fft,
                                          win_length=win_length_samples,
                                          hop_length=hop_length_samples,
                                          center=center,
                                          window=window))
        elif spectrogram_type == 'power':
            return numpy.abs(librosa.stft(y + self.eps,
                                          n_fft=n_fft,
                                          win_length=win_length_samples,
                                          hop_length=hop_length_samples,
                                          center=center,
                                          window=window)) ** 2
        else:
            message = '{name}: Unknown spectrum type [{spectrogram_type}]'.format(
                name=self.__class__.__name__,
                spectrogram_type=spectrogram_type
            )

            self.logger.exception(message)
            raise ValueError(message)

    def _update(self, d, u):
        """Recursive dict update
        """

        for k, v in iteritems(u):
            if isinstance(v, collections.Mapping):
                r = self._update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d


class FeatureProcessingUnitMixin(object):
    """Feature processing chain unit mixin"""
    def process(self, feature_data):
        pass


class FeatureStacker(FeatureProcessingUnitMixin):
    """Feature stacker"""
    __version__ = '0.0.1'

    def __init__(self, recipe, feature_hop=1, **kwargs):
        """Constructor

        Parameters
        ----------
        recipe : dict
            Stacking recipe
        feature_hop : int, optional
            Feature hopping
            Default value 1

        """

        self.recipe = recipe
        self.feature_hop = feature_hop
        self.logger = kwargs.get('logger', logging.getLogger(__name__))

    def __getstate__(self):
        """Return only needed data for pickle"""
        return {
            'recipe': self.recipe,
            'feature_hop': self.feature_hop,
        }

    def __setstate__(self, d):
        self.recipe = d['recipe']
        self.feature_hop = d['feature_hop']
        self.logger = logging.getLogger(__name__)

    def normalizer(self, normalizer_list):
        """Stack normalization factors based on stack map

        Parameters
        ----------
        normalizer_list : dict
            List of Normalizer classes

        Returns
        -------
        dict
            Stacked normalizer variables in a dict

        """

        # Check that all feature matrices have same amount of frames
        frame_count = -1
        for feature in self.recipe:
            method = feature['method']
            if 'vector-index' in feature:
                channel = feature['vector-index']['channel']

            else:
                channel = 0  # Default value

            if frame_count == -1:
                frame_count = normalizer_list[method]['N']

            elif frame_count != normalizer_list[method]['N']:
                message = '{name}: Normalizers should have seen same number of frames {count1} != {count2} [{method}]'.format(
                    name=self.__class__.__name__,
                    count1=frame_count,
                    count2=normalizer_list[method]['N'],
                    method=method)

                self.logger.exception(message)
                raise AssertionError(message)

        stacked_mean = []
        stacked_std = []

        for feature in self.recipe:
            method = feature['method']

            # Default values
            channel = 0
            if 'vector-index' in feature:
                channel = feature['vector-index']['channel']

            if ('vector-index' not in feature or
               ('vector-index' in feature and 'full' in feature['vector-index'] and feature['vector-index']['full'])):

                # We have Full matrix
                stacked_mean.append(normalizer_list[method]['mean'][channel])
                stacked_std.append(normalizer_list[method]['std'][channel])

            elif ('vector-index' in feature and
                  'vector' in feature['vector-index'] and
                  'selection' in feature['vector-index'] and feature['vector-index']['selection']):

                # We have selector vector
                stacked_mean.append(normalizer_list[method]['mean'][channel][:, feature['vector-index']['vector']])
                stacked_std.append(normalizer_list[method]['std'][channel][:, feature['vector-index']['vector']])

            elif ('vector-index' in feature and
                  'start' in feature['vector-index'] and
                  'end' in feature['vector-index']):

                # we have start and end index
                stacked_mean.append(normalizer_list[method]['mean'][channel][:, feature['vector-index']['start']:feature['vector-index']['end']])
                stacked_std.append(normalizer_list[method]['std'][channel][:, feature['vector-index']['start']:feature['vector-index']['end']])

        normalizer = {
            'mean': [numpy.hstack(stacked_mean)],
            'std': [numpy.hstack(stacked_std)],
            'N': [frame_count],
        }

        return normalizer

    def feature_vector(self, feature_repository):
        """Feature vector creation

        Parameters
        ----------
        feature_repository : FeatureRepository, dict
            Feature repository with needed features

        Returns
        -------
        FeatureContainer

        """

        # Check that all feature matrices have same amount of frames
        frame_count = -1
        for feature in self.recipe:
            method = feature['method']
            channel = 0  # Default value
            if 'vector-index' in feature:
                channel = feature['vector-index']['channel']

            if frame_count == -1:
                frame_count = feature_repository[method].feat[channel].shape[0]

            elif frame_count != feature_repository[method].feat[channel].shape[0]:
                message = '{name}: Feature matrices should have same number of frames {count1} != {count2} [{method}]'.format(
                    name=self.__class__.__name__,
                    count1=frame_count,
                    count2=feature_repository[method].feat[channel].shape[0],
                    method=method
                )

                self.logger.exception(message)
                raise AssertionError(message)

        # Stack features
        feature_matrix = []
        for feature in self.recipe:
            method = feature['method']

            # Default values
            channel = 0
            if 'vector-index' in feature:
                channel = feature['vector-index']['channel']

            if ('vector-index' not in feature or
               ('vector-index' in feature and 'full' in feature['vector-index'] and feature['vector-index']['full'])):

                # We have Full matrix
                feature_matrix.append(feature_repository[method].feat[channel][::self.feature_hop, :])

            elif ('vector-index' in feature and
                  'vector' in feature['vector-index'] and
                  'selection' in feature['vector-index'] and feature['vector-index']['selection']):

                index = numpy.array(feature['vector-index']['vector'])
                # We have selector vector
                feature_matrix.append(feature_repository[method].feat[channel][::self.feature_hop, index])

            elif ('vector-index' in feature and
                  'start' in feature['vector-index'] and
                  'end' in feature['vector-index']):

                # we have start and end index
                feature_matrix.append(feature_repository[method].feat[channel][::self.feature_hop, feature['vector-index']['start']:feature['vector-index']['end']])

        meta = {
            'parameters': {
                'fs': feature_repository[method].meta['parameters']['fs'],
                'win_length_seconds': feature_repository[method].meta['parameters'].get('win_length_seconds'),
                'win_length_samples': feature_repository[method].meta['parameters'].get('win_length_samples'),
                'hop_length_seconds': feature_repository[method].meta['parameters'].get('hop_length_seconds'),
                'hop_length_samples': feature_repository[method].meta['parameters'].get('hop_length_samples'),
            },
            'datetime': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            'audio_file': feature_repository[method].meta['audio_file'],
            'extractor_version': None,
        }

        return FeatureContainer(features=[numpy.hstack(feature_matrix)], meta=meta)

    def process(self, feature_data):
        """Feature vector creation

        Parameters
        ----------
        feature_data : FeatureRepository
            Feature repository with needed features

        Returns
        -------
        FeatureContainer

        """

        return self.feature_vector(feature_repository=feature_data)


class FeatureNormalizer(DataFile, ContainerMixin, FeatureProcessingUnitMixin):
    """Feature normalizer

    Accumulates feature statistics

    Examples
    --------

    >>> normalizer = FeatureNormalizer()
    >>> for feature_matrix in training_items:
    >>>     normalizer.accumulate(feature_matrix)
    >>>
    >>> normalizer.finalize()

    >>> for feature_matrix in test_items:
    >>>     feature_matrix_normalized = normalizer.normalize(feature_matrix)
    >>>     # used the features

    """
    __version__ = '0.0.1'

    def __init__(self, stat=None, feature_matrix=None):
        """__init__ method.

        Parameters
        ----------
        stat : dict or None
            Pre-calculated statistics in dict to initialize internal state

        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)] or None
            Feature matrix to be used in the initialization

        """

        if stat:
            defaults = {
                    'N': [],
                    'S1': [],
                    'S2': [],
                    'mean': [],
                    'std': [],
                }
            defaults.update(stat)
            super(DataFile, self).__init__(defaults)

        elif feature_matrix and stat is None:
            super(DataFile, self).__init__(
                {
                    'N': [feature_matrix.shape[0]],
                    'S1': [numpy.sum(feature_matrix, axis=0)],
                    'S2': [numpy.sum(feature_matrix ** 2, axis=0)],
                    'mean': [numpy.mean(feature_matrix, axis=0)],
                    'std': [numpy.std(feature_matrix, axis=0)],
                }
            )
            self.finalize()
        else:
            super(DataFile, self).__init__(
                {
                    'N': [],
                    'S1': [],
                    'S2': [],
                    'mean': [],
                    'std': [],
                }
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Finalize accumulated calculation
        self.finalize()

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'N': self['N'],
            'S1': self['S1'],
            'S2': self['S2'],
            'mean': self['mean'],
            'std': self['std'],
        }

    def __setstate__(self, d):
        self.N = d['N']
        self.S1 = d['S1']
        self.S2 = d['S2']
        self.mean = d['mean']
        self.std = d['std']

    def accumulate(self, feature_container):
        """Accumulate statistics

        Parameters
        ----------
        feature_container : FeatureContainer

        Returns
        -------
        nothing

        """

        stat = feature_container.stat
        for channel in range(0, len(stat)):
            if len(self['N']) <= channel:
                self['N'].insert(channel, 0)

            self['N'][channel] += stat[channel]['N']

            if len(self['mean']) <= channel:
                self['mean'].insert(channel, 0)
            self['mean'][channel] += stat[channel]['mean']

            if len(self['S1']) <= channel:
                self['S1'].insert(channel, 0)
            self['S1'][channel] += stat[channel]['S1']

            if len(self['S2']) <= channel:
                self['S2'].insert(channel, 0)
            self['S2'][channel] += stat[channel]['S2']
        return self

    def finalize(self):
        """Finalize statistics calculation

        Accumulated values are used to get mean and std for the seen feature data.

        Parameters
        ----------

        Returns
        -------
        None

        """

        for channel in range(0, len(self['N'])):
            # Finalize statistics
            self['mean'][channel] = self['S1'][channel] / self['N'][channel]

            if len(self['std']) <= channel:
                self['std'].insert(channel, 0)
            self['std'][channel] = numpy.sqrt((self['N'][channel] * self['S2'][channel] - (self['S1'][channel] * self['S1'][channel])) / (self['N'][channel] * (self['N'][channel] - 1)))

            # In case we have very brain-death material we get std = Nan => 0.0
            self['std'][channel] = numpy.nan_to_num(self['std'][channel])

            self['mean'][channel] = numpy.reshape(self['mean'][channel], [1, -1])
            self['std'][channel] = numpy.reshape(self['std'][channel], [1, -1])
        return self

    def normalize(self, feature_container, channel=0):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_container : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized
        channel : int
            Feature channel
            Default value "0"

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """

        if isinstance(feature_container, FeatureContainer):
            feature_container.feat[channel] = (feature_container.feat[channel] - self['mean'][channel]) / self['std'][channel]
            return feature_container

        elif isinstance(feature_container, numpy.ndarray):
            return (feature_container - self['mean'][channel]) / self['std'][channel]

    def process(self, feature_data):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_data : FeatureContainer or numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """

        return self.normalize(feature_container=feature_data)


class FeatureAggregator(FeatureProcessingUnitMixin):
    """Feature aggregator"""
    __version__ = '0.0.1'

    valid_method = ['mean', 'std', 'cov', 'kurtosis', 'skew', 'flatten']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        recipe : list of dict or list of str
            Aggregation recipe, supported methods [mean, std, cov, kurtosis, skew, flatten].
        win_length_frames : int
            Window length in feature frames
        hop_length_frames : int
            Hop length in feature frames

        """

        if isinstance(kwargs.get('recipe'), dict):
            self.recipe = [d['method'] for d in kwargs.get('recipe')]
        elif isinstance(kwargs.get('recipe'), list):
            recipe = kwargs.get('recipe')
            if isinstance(recipe[0], dict):
                self.recipe = [d['method'] for d in kwargs.get('recipe')]
            else:
                self.recipe = recipe

        self.win_length_frames = kwargs.get('win_length_frames')
        self.hop_length_frames = kwargs.get('hop_length_frames')

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'recipe': self.recipe,
            'win_length_frames': self.win_length_frames,
            'hop_length_frames': self.hop_length_frames,
        }

    def __setstate__(self, d):
        self.recipe = d['recipe']
        self.win_length_frames = d['win_length_frames']
        self.hop_length_frames = d['hop_length_frames']

    def process(self, feature_data):
        """Process features

        Parameters
        ----------
        feature_data : FeatureContainer
            Features to be aggregated
        Returns
        -------
        FeatureContainer

        """

        # Not the most efficient way as numpy stride_tricks would produce
        # faster code, however, opted for cleaner presentation this time.
        feature_data_per_channel = []
        for channel in range(0, feature_data.channels):
            aggregated_features = []
            for frame in range(0, feature_data.feat[channel].shape[0], self.hop_length_frames):
                # Get start and end of the window, keep frame at the middle (approximately)
                start_frame = int(frame - numpy.floor(self.win_length_frames/2.0))
                end_frame = int(frame + numpy.ceil(self.win_length_frames / 2.0))

                frame_id = numpy.array(range(start_frame, end_frame))
                # If start of feature matrix, pad with first frame
                frame_id[frame_id < 0] = 0

                # If end of the feature matrix, pad with last frame
                frame_id[frame_id > feature_data.feat[channel].shape[0] - 1] = feature_data.feat[channel].shape[0] - 1

                current_frame = feature_data.feat[channel][frame_id, :]
                aggregated_frame = []

                if 'mean' in self.recipe:
                    aggregated_frame.append(current_frame.mean(axis=0))

                if 'std' in self.recipe:
                    aggregated_frame.append(current_frame.std(axis=0))

                if 'cov' in self.recipe:
                    aggregated_frame.append(numpy.cov(current_frame).flatten())

                if 'kurtosis' in self.recipe:
                    aggregated_frame.append(scipy.stats.kurtosis(current_frame))

                if 'skew' in self.recipe:
                    aggregated_frame.append(scipy.stats.skew(current_frame))

                if 'flatten' in self.recipe:
                    aggregated_frame.append(current_frame.flatten())

                if aggregated_frame:
                    aggregated_features.append(numpy.concatenate(aggregated_frame))

            feature_data_per_channel.append(numpy.vstack(aggregated_features))

        meta = {
            'parameters': {
                'recipe': self.recipe,
                'win_length_frames': self.win_length_frames,
                'hop_length_frames': self.hop_length_frames,
            },
            'datetime': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        }

        if 'audio_file' in feature_data.meta:
            meta['audio_file'] = feature_data.meta['audio_file']

        return FeatureContainer(features=feature_data_per_channel, meta=meta)


class FeatureMasker(object):
    """Feature masker"""
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        hop_length_seconds : float
            Hop length in seconds

        """
        self.hop_length_seconds = kwargs.get('hop_length_seconds')

        # Initialize mask events
        self.mask_events = MetaDataContainer()

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'hop_length_seconds': self.hop_length_seconds,
        }

    def __setstate__(self, d):
        self.hop_length_seconds = d['hop_length_seconds']
        self.mask_events = MetaDataContainer()

    def set_mask(self, mask_events):
        """Set masking events

        Parameters
        ----------
        mask_events : list of MetaItems or MetaDataContainer
            Event list used for masking

        """

        self.mask_events = mask_events
        return self

    def masking(self, feature_data, mask_event):
        """Masking feature repository with given events

        Parameters
        ----------
        feature_data : FeatureRepository
        mask_events : list of MetaItems or MetaDataContainer
            Event list used for masking

        Returns
        -------
        FeatureRepository

        """

        for method in list(feature_data.keys()):
            removal_mask = numpy.ones((feature_data[method].shape[0]), dtype=bool)
            for mask_event in self.mask_events:
                onset_frame = int(numpy.floor(mask_event.event_onset / self.hop_length_seconds))
                offset_frame = int(numpy.ceil(mask_event.event_offset / self.hop_length_seconds))
                if offset_frame > feature_data[method].shape[0]:
                    offset_frame = feature_data[method].shape[0]
                removal_mask[onset_frame:offset_frame] = False

            for channel in range(0, feature_data[method].channels):
                feature_data[method].feat[channel] = feature_data[method].feat[channel][removal_mask, :]

        return feature_data

    def process(self, feature_data):
        """Process feature repository

        Parameters
        ----------
        feature_data : FeatureRepository

        Returns
        -------
        FeatureRepository

        """

        if self.mask_events:
            return self.masking(feature_data=feature_data, mask_event=self.mask_events)
        else:
            return feature_data

