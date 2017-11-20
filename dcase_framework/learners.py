#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learners
========
Classes for machine learning

SceneClassifier
^^^^^^^^^^^^^^^

SceneClassifierGMM
..................

Scene classifier with GMM. This learner is using ``sklearn.mixture.GaussianMixture`` implementation. See
`documentation <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html/>`_.

.. autosummary::
    :toctree: generated/

    SceneClassifierGMM
    SceneClassifierGMM.learn
    SceneClassifierGMM.predict

SceneClassifierMLP
..................

Scene classifier with MLP. This learner is a simple MLP based learner using Keras neural network implementation
and sequential API. See `documentation <https://keras.io/>`_.

.. autosummary::
    :toctree: generated/

    SceneClassifierMLP
    SceneClassifierMLP.learn
    SceneClassifierMLP.predict

SceneClassifierKerasSequential
..............................

Scene classifier with Keras sequential API (see `documentation <https://keras.io/>`_). This learner can be used for
more advanced network structures than SceneClassifierMLP.

.. autosummary::
    :toctree: generated/

    SceneClassifierKerasSequential
    SceneClassifierKerasSequential.learn
    SceneClassifierKerasSequential.predict

EventDetector
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    EventDetector

EventDetectorGMM
................

.. autosummary::
    :toctree: generated/

    EventDetectorGMM
    EventDetectorGMM.learn
    EventDetectorGMM.predict

EventDetectorMLP
................

.. autosummary::
    :toctree: generated/

    EventDetectorMLP
    EventDetectorMLP.learn
    EventDetectorMLP.predict

EventDetectorKerasSequential
............................

.. autosummary::
    :toctree: generated/

    EventDetectorKerasSequential
    EventDetectorKerasSequential.learn
    EventDetectorKerasSequential.predict

LearnerContainer - Base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    LearnerContainer
    LearnerContainer.class_labels
    LearnerContainer.method
    LearnerContainer.params
    LearnerContainer.feature_masker
    LearnerContainer.feature_normalizer
    LearnerContainer.feature_stacker
    LearnerContainer.feature_aggregator
    LearnerContainer.model
    LearnerContainer.set_seed
    LearnerContainer.learner_params

"""

from __future__ import print_function, absolute_import
from six import iteritems

import sys
import numpy
import logging
import random
import warnings
import copy

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from .files import DataFile
from .containers import ContainerMixin, DottedDict
from .features import FeatureContainer
from .utils import SuppressStdoutAndStderr
from .metadata import MetaDataItem, EventRoll
from .keras_utils import KerasMixin, BaseDataGenerator, StasherCallback
from .data import DataSequencer
from .utils import get_class_inheritors
from .recognizers import SceneRecognizer, EventRecognizer


def scene_classifier_factory(*args, **kwargs):
    if kwargs.get('method', None) == 'gmm':
        return SceneClassifierGMM(*args, **kwargs)
    elif kwargs.get('method', None) == 'mlp':
        return SceneClassifierMLP(*args, **kwargs)
    else:
        raise ValueError('{name}: Invalid SegmentClassifier method [{method}]'.format(
            name='segment_classifier_factory',
            method=kwargs.get('method', None))
        )


def event_detector_factory(*args, **kwargs):
    if kwargs.get('method', None) == 'gmm':
        return EventDetectorGMM(*args, **kwargs)

    elif kwargs.get('method', None) == 'mlp':
        return EventDetectorMLP(*args, **kwargs)

    elif kwargs.get('method', None) == 'keras_seq':
        return EventDetectorKerasSequential(*args, **kwargs)

    else:
        raise ValueError('{name}: Invalid EventDetector method [{method}]'.format(
            name='event_detector_factory',
            method=kwargs.get('method', None))
        )


class LearnerContainer(DataFile, ContainerMixin):
    valid_formats = ['cpickle']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        method : str
            Method label
            Default value "None"
        class_labels : list of strings
            List of class labels
            Default value "[]"
        params : dict or DottedDict
            Parameters
        feature_masker : FeatureMasker or class inherited from FeatureMasker
            Feature masker instance
            Default value "None"
        feature_normalizer : FeatureNormalizer or class inherited from FeatureNormalizer
            Feature normalizer instance
            Default value "None"
        feature_stacker : FeatureStacker or class inherited from FeatureStacker
            Feature stacker instance
            Default value "None"
        feature_aggregator : FeatureAggregator or class inherited from FeatureAggregator
            Feature aggregator instance
            Default value "None"
        logger : logging
            Instance of logging
            Default value "None"
        disable_progress_bar : bool
            Disable progress bar in console
            Default value "False"
        log_progress : bool
            Show progress in log.
            Default value "False"
        show_extra_debug : bool
            Show extra debug information
            Default value "True"

        """

        super(LearnerContainer, self).__init__({
            'method': kwargs.get('method', None),
            'class_labels': kwargs.get('class_labels', []),
            'params': DottedDict(kwargs.get('params', {})),
            'feature_masker': kwargs.get('feature_masker', None),
            'feature_normalizer': kwargs.get('feature_normalizer', None),
            'feature_stacker': kwargs.get('feature_stacker', None),
            'feature_aggregator': kwargs.get('feature_aggregator', None),
            'model': kwargs.get('model', {}),
            'learning_history': kwargs.get('learning_history', {}),
        }, *args, **kwargs)

        # Set randomization seed
        if self.params.get_path('seed') is not None:
            self.seed = self.params.get_path('seed')
        elif self.params.get_path('parameters.seed') is not None:
            self.seed = self.params.get_path('parameters.seed')
        elif kwargs.get('seed', None):
            self.seed = kwargs.get('seed')
        else:
            epoch = datetime.utcfromtimestamp(0)
            unix_now = (datetime.now() - epoch).total_seconds() * 1000.0
            bigint, mod = divmod(int(unix_now) * 1000, 2**32)
            self.seed = mod

        self.logger = kwargs.get('logger',  logging.getLogger(__name__))
        self.disable_progress_bar = kwargs.get('disable_progress_bar',  False)
        self.log_progress = kwargs.get('log_progress',  False)
        self.show_extra_debug = kwargs.get('show_extra_debug', True)

    @property
    def class_labels(self):
        """Class labels

        Returns
        -------
        list of strings
            List of class labels in the model
        """
        return sorted(self.get('class_labels', None))

    @class_labels.setter
    def class_labels(self, value):
        self['class_labels'] = value

    @property
    def method(self):
        """Learner method label

        Returns
        -------
        str
            Learner method label
        """

        return self.get('method', None)

    @method.setter
    def method(self, value):
        self['method'] = value

    @property
    def params(self):
        """Parameters

        Returns
        -------
        DottedDict
            Parameters
        """
        return self.get('params', None)

    @params.setter
    def params(self, value):
        self['params'] = value

    @property
    def feature_masker(self):
        """Feature masker instance

        Returns
        -------
        FeatureMasker

        """

        return self.get('feature_masker', None)

    @feature_masker.setter
    def feature_masker(self, value):
        self['feature_masker'] = value

    @property
    def feature_normalizer(self):
        """Feature normalizer instance

        Returns
        -------
        FeatureNormalizer

        """

        return self.get('feature_normalizer', None)

    @feature_normalizer.setter
    def feature_normalizer(self, value):
        self['feature_normalizer'] = value

    @property
    def feature_stacker(self):
        """Feature stacker instance

        Returns
        -------
        FeatureStacker

        """

        return self.get('feature_stacker', None)

    @feature_stacker.setter
    def feature_stacker(self, value):
        self['feature_stacker'] = value

    @property
    def feature_aggregator(self):
        """Feature aggregator instance

        Returns
        -------
        FeatureAggregator

        """

        return self.get('feature_aggregator', None)

    @feature_aggregator.setter
    def feature_aggregator(self, value):
        self['feature_aggregator'] = value

    @property
    def model(self):
        """Acoustic model

        Returns
        -------
        model

        """

        return self.get('model', None)

    @model.setter
    def model(self, value):
        self['model'] = value

    def set_seed(self, seed=None):
        """Set randomization seeds

        Returns
        -------
        nothing

        """

        if seed is None:
            seed = self.seed

        numpy.random.seed(seed)
        random.seed(seed)

    @property
    def learner_params(self):
        """Get learner parameters from parameter container

        Returns
        -------
        DottedDict
            Learner parameters

        """

        if 'parameters' in self['params']:
            parameters = self['params']['parameters']
        else:
            parameters = self['params']

        return DottedDict({k: v for k, v in parameters.items() if not k.startswith('_')})

    def _get_input_size(self, data):
        input_shape = None
        for audio_filename in data:
            if not input_shape:
                input_shape = data[audio_filename].feat[0].shape[1]
            elif input_shape != data[audio_filename].feat[0].shape[1]:
                message = '{name}: Input size not coherent.'.format(
                    name=self.__class__.__name__
                )
                self.logger.exception(message)
                raise ValueError(message)

        return input_shape


class SceneClassifier(LearnerContainer):
    """Scene classifier (Frame classifier / Multi-class - Single-label)
    """

    def predict(self, feature_data):
        """Predict frame probabilities for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
            Feature data

        Returns
        -------
        str
            class label

        """

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame probabilities
        return self._frame_probabilities(feature_data)

    def _generate_validation(self, annotations, validation_type='generated_scene_balanced',
                             valid_percentage=0.20, seed=None):
        self.set_seed(seed=seed)
        validation_files = []

        if validation_type == 'generated_scene_balanced':
            # Get training data per scene label
            annotation_data = {}
            for audio_filename in sorted(list(annotations.keys())):
                scene_label = annotations[audio_filename]['scene_label']
                location_id = annotations[audio_filename]['identifier']
                if scene_label not in annotation_data:
                    annotation_data[scene_label] = {}
                if location_id not in annotation_data[scene_label]:
                    annotation_data[scene_label][location_id] = []
                annotation_data[scene_label][location_id].append(audio_filename)

            training_files = []
            validation_amounts = {}

            for scene_label in sorted(annotation_data.keys()):
                validation_amount = []
                sets_candidates = []
                for i in range(0, 1000):
                    current_locations = list(annotation_data[scene_label].keys())
                    random.shuffle(current_locations, random.random)
                    valid_percentage_index = int(numpy.ceil(valid_percentage * len(annotation_data[scene_label])))
                    current_validation_locations = current_locations[0:valid_percentage_index]
                    current_training_locations = current_locations[valid_percentage_index:]

                    # Collect validation files
                    current_validation_files = []
                    for location_id in current_validation_locations:
                        current_validation_files += annotation_data[scene_label][location_id]

                    # Collect training files
                        current_training_files = []
                    for location_id in current_training_locations:
                        current_training_files += annotation_data[scene_label][location_id]

                    validation_amount.append(
                        len(current_validation_files) / float(len(current_validation_files) + len(current_training_files))
                    )

                    sets_candidates.append({
                        'validation': current_validation_files,
                        'training': current_training_files,
                    })

                best_set_id = numpy.argmin(numpy.abs(numpy.array(validation_amount) - valid_percentage))
                validation_files += sets_candidates[best_set_id]['validation']
                training_files += sets_candidates[best_set_id]['training']
                validation_amounts[scene_label] = validation_amount[best_set_id]

            if self.show_extra_debug:
                self.logger.debug('  Validation set statistics')
                self.logger.debug('  {0:<20s} | {1:10s} '.format('Scene label', 'Validation amount (%)'))
                self.logger.debug('  {0:<20s} + {1:10s} '.format('-'*20, '-'*20))

                for scene_label in sorted(validation_amounts.keys()):
                    self.logger.debug('  {0:<20s} | {1:4.2f} '.format(scene_label, validation_amounts[scene_label]*100))
                self.logger.debug('  ')

        else:
            message = '{name}: Unknown validation_type [{type}].'.format(
                name=self.__class__.__name__,
                type=validation_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

        return validation_files

    def _frame_probabilities(self, feature_data):
        # Implement in child class
        pass

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            frame_count = data[audio_filename].feat[0].shape[0]
            pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
            roll = numpy.zeros((frame_count, len(self.class_labels)))
            roll[:, pos] = 1
            activity_matrix_dict[audio_filename] = roll
        return activity_matrix_dict

    def learn(self, data, annotations, data_filenames=None):
        message = '{name}: Implement learn function.'.format(
            name=self.__class__.__name__
        )

        self.logger.exception(message)
        raise AssertionError(message)


class SceneClassifierGMM(SceneClassifier):
    """Scene classifier with GMM

    This learner is using ``sklearn.mixture.GaussianMixture`` implementation. See
    `documentation <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html/>`_.

    Usage example:

    .. code-block:: python
        :linenos:

        # Audio files
        files = ['example1.wav', 'example2.wav', 'example3.wav']

        # Meta data
        annotations = {
            'example1.wav': MetaDataItem(
                {
                    'file': 'example1.wav',
                    'scene_label': 'SceneA'
                }
            ),
            'example2.wav':MetaDataItem(
                {
                    'file': 'example2.wav',
                    'scene_label': 'SceneB'
                }
            ),
            'example3.wav': MetaDataItem(
                {
                    'file': 'example3.wav',
                    'scene_label': 'SceneC'
                }
            ),
        }

        # Extract features
        feature_data = {}
        for file in files:
            feature_data[file] = FeatureExtractor().extract(
                audio_file=file,
                extractor_name='mfcc',
                extractor_params={
                    'mfcc': {
                        'n_mfcc': 10
                    }
                }
            )['mfcc']

        # Learn acoustic model
        learner_params = {
            'n_components': 1,
            'covariance_type': 'diag',
            'tol': 0.001,
            'reg_covar': 0,
            'max_iter': 40,
            'n_init': 1,
            'init_params': 'kmeans',
            'random_state': 0,
        }

        gmm_learner = SceneClassifierGMM(
            filename='gmm_model.cpickle',
            class_labels=['SceneA', 'SceneB', 'SceneC'],
            params=learner_params,
        )

        gmm_learner.learn(
            data=feature_data,
            annotations=annotations
        )

        # Recognition
        recognizer_params = {
            'frame_accumulation': {
                'enable': True,
                'type': 'sum'
            },
            'decision_making': {
                'enable': True,
                'type': 'maximum',
            }
        }
        correctly_predicted = 0
        for file in feature_data:
            frame_probabilities = gmm_learner.predict(
                feature_data=feature_data[file],
            )

            # Scene recognizer
            current_result = SceneRecognizer(
                params=recognizer_params,
                class_labels=gmm_learner.class_labels,
            ).process(
                frame_probabilities=frame_probabilities
            )

            if annotations[file].scene_label == current_result:
                correctly_predicted += 1
            print(current_result, annotations[file].scene_label)

        print('Accuracy = {:3.2f} %'.format(correctly_predicted/float(len(feature_data))*100))

    **Learner parameters**

    +--------------------------------+--------------+------------------------------------------------------------------+
    | Field name                     | Value type   | Description                                                      |
    +================================+==============+==================================================================+
    | n_components                   | int          | The number of mixture components.                                |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | covariance_type                | string       | Covariance type.                                                 |
    |                                | { full |     |                                                                  |
    |                                | tied |       |                                                                  |
    |                                | diag |       |                                                                  |
    |                                | spherical }  |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | tol                            | float        | Covariance threshold.                                            |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | reg_covar                      | float        | Non-negative regularization added to the diagonal of covariance. |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | max_iter                       | int          | The number of EM iterations.                                     |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | n_init                         | int          | The number of initializations.                                   |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | init_params                    | string       | The method used to initialize model weights.                     |
    |                                | { kmeans |   |                                                                  |
    |                                | random }     |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | random_state                   | int          | Random seed.                                                     |
    +--------------------------------+--------------+------------------------------------------------------------------+

    """

    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'show_model_information': False,
            'audio_error_handling': False,
            'win_length_seconds': 0.04,
            'hop_length_seconds': 0.02,
            'method': 'gmm',
            'parameters': {
                'covariance_type': 'diag',
                'init_params': 'kmeans',
                'max_iter': 40,
                'n_components': 16,
                'n_init': 1,
                'random_state': 0,
                'reg_covar': 0,
                'tol': 0.001
            },
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(SceneClassifierGMM, self).__init__(*args, **kwargs)
        self.method = 'gmm'

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data

        Returns
        -------
        self

        """

        if self.learner_params.get_path('validation.enable', False):
            message = '{name}: Validation is not implemented for this learner.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)

        from sklearn.mixture import GaussianMixture

        training_files = sorted(list(annotations.keys()))  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        class_progress = tqdm(self.class_labels,
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              miniters=1,
                              disable=self.disable_progress_bar
                              )

        for class_id, class_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {class_label:<15s}'.format(
                    title='Learn',
                    item_id=class_id,
                    total=len(self.class_labels),
                    class_label=class_label)
                )
            current_class_data = X_training[Y_training[:, class_id] > 0, :]

            self['model'][class_label] = GaussianMixture(**self.learner_params).fit(current_class_data)

        return self

    def _frame_probabilities(self, feature_data):
        logls = numpy.ones((len(self['model']), feature_data.shape[0])) * -numpy.inf

        for label_id, label in enumerate(self.class_labels):
            logls[label_id] = self['model'][label].score(feature_data)

        return logls


class SceneClassifierGMMdeprecated(SceneClassifier):
    """Scene classifier with GMM"""
    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'show_model_information': False,
            'audio_error_handling': False,
            'win_length_seconds': 0.04,
            'hop_length_seconds': 0.02,
            'method': 'gmm_deprecated',
            'parameters': {
                'n_components': 16,
                'covariance_type': 'diag',
                'random_state': 0,
                'tol': 0.001,
                'min_covar': 0.001,
                'n_iter': 40,
                'n_init': 1,
                'params': 'wmc',
                'init_params': 'wmc',
            },
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(SceneClassifierGMMdeprecated, self).__init__(*args, **kwargs)
        self.method = 'gmm_deprecated'

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data

        Returns
        -------
        self

        """

        if self.learner_params.get_path('validation.enable', False):
            message = '{name}: Validation is not implemented for this learner.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        training_files = sorted(list(annotations.keys()))  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        class_progress = tqdm(self.class_labels,
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              miniters=1,
                              disable=self.disable_progress_bar
                              )

        for class_id, class_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {class_label:<15s}'.format(
                    title='Learn',
                    item_id=class_id,
                    total=len(self.class_labels),
                    class_label=class_label)
                )

            current_class_data = X_training[Y_training[:, class_id] > 0, :]

            self['model'][class_label] = mixture.GMM(**self.learner_params).fit(current_class_data)

        return self

    def _frame_probabilities(self, feature_data):
        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        logls = numpy.ones((len(self['model']), feature_data.shape[0])) * -numpy.inf

        for label_id, label in enumerate(self.class_labels):
            logls[label_id] = self['model'][label].score(feature_data)

        return logls


class SceneClassifierMLP(SceneClassifier, KerasMixin):
    """Scene classifier with MLP

    This learner is a simple MLP based learner using Keras neural network implementation and sequential API.
    See `documentation <https://keras.io/>`_.

    **Learner parameters**

    +--------------------------------+--------------+------------------------------------------------------------------+
    | Field name                     | Value type   | Description                                                      |
    +================================+==============+==================================================================+
    | seed                           | int          | Randomization seed. Use this to make learner behaviour           |
    |                                |              | deterministic.                                                   |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **keras**                                                                                                        |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | backend                        | string       | Keras backend selector.                                          |
    |                                | {theano |    |                                                                  |
    |                                | tensorflow}  |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **keras->backend_parameters**                                                                                    |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | device                         | string       | Device selector. ``cpu`` is best option to produce deterministic |
    |                                | {cpu | gpu}  | results. All baseline results are calculated in cpu mode.        |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | floatX                         | string       | Float number type. Usually float32 used since that is compatible |
    |                                |              | with GPUs. Valid only for ``theano`` backend.                    |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | fastmath                       | bool         | If true, will enable fastmath mode when CUDA code is compiled.   |
    |                                |              | Div and sqrt are faster, but precision is lower. This can cause  |
    |                                |              | numerical issues some in cases. Valid only for ``theano``        |
    |                                |              | backend and GPU mode.                                            |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | optimizer                      | string       | Compilation mode for theano functions.                           |
    |                                | {fast_run |  |                                                                  |
    |                                | merge |      |                                                                  |
    |                                | fast_compile |                                                                  |
    |                                | None}        |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | openmp                         | bool         | If true, Theano will use multiple cores, see `more               |
    |                                |              | <http://deeplearning.net/software/theano/                        |
    |                                |              | tutorial/multi_cores.html>`_                                     |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | threads                        | int          | Number of threads used. Use one to disable threading.            |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | CNR                            | bool         | Conditional numerical reproducibility for MKL BLAS. When set to  |
    |                                |              | True, compatible mode used.                                      |
    |                                |              | See `more <https://software.intel.com/en-us/node/528408>`_.      |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **validation**                                                                                                   |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | enable                         | bool         | If true, validation set is used during the training procedure.   |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | setup_source                   | string       | Validation setup source. Valid sources:                          |
    |                                |              |                                                                  |
    |                                |              | - ``generated_scene_balanced``, balanced based on scene labels,  |
    |                                |              |   used for Task1.                                                |
    |                                |              | - ``generated_event_file_balanced``, balanced based on events,   |
    |                                |              |   used for Task2.                                                |
    |                                |              | - ``generated_scene_location_event_balanced``, balanced          |
    |                                |              |   based on scene, location and events. Used for Task3.           |
    |                                |              |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | validation_amount              | float        | Percentage of training data selected for validation. Use value   |
    |                                |              | between 0.0-1.0.                                                 |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | seed                           | int          | Validation set generation seed. If None, learner seed will be    |
    |                                |              | used.                                                            |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **training**                                                                                                     |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | epochs                         | int          | Number of epochs.                                                |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | batch_size                     | int          | Batch size.                                                      |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | shuffle                        | bool         | If true, training samples are shuffled at each epoch.            |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **training->callbacks**, list of parameter sets in following format. Callback called during the model training.  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | type                           | string       | Callback name, use standard keras callbacks                      |
    |                                |              | `callbacks <https://keras.io/callbacks/>`_ or ones defined by    |
    |                                |              | dcase_framework (Plotter, Stopper, Stasher).                     |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | parameters                     | dict         | Place inside this all parameters for the callback.               |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **training->model->config**, list of dicts. Defining network topology.                                           |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | class_name                     | string       | Layer name. Use standard keras                                   |
    |                                |              | `core layers <https://keras.io/layers/core/>`_,                  |
    |                                |              | `convolutional                                                   |
    |                                |              | layers <https://keras.io/layers/convolutional/>`_,               |
    |                                |              | `pooling layers <https://keras.io/layers/pooling/>`_,            |
    |                                |              | `recurrent layers <https://keras.io/layers/recurrent/>`_, or     |
    |                                |              | `normalization layers <https://keras.io/layers/normalization/>`_ |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | config                         | dict         | Place inside this all parameters for the layer.                  |
    |                                |              | See Keras documentation. Magic parameter values:                 |
    |                                |              |                                                                  |
    |                                |              | - ``FEATURE_VECTOR_LENGTH``, feature vector length.              |
    |                                |              |   This automatically inserted for input layer.                   |
    |                                |              | - ``CLASS_COUNT``, number of classes.                            |
    |                                |              |                                                                  |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | input_shape                    | list of      | List of integers which is converted into tuple before giving to  |
    |                                | ints         | Keras layer.                                                     |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **training->model**                                                                                              |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | loss                           | string       | Keras loss function name. See                                    |
    |                                |              | `Keras documentation <https://keras.io/losses/>`_.               |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | metrics                        | list of      | Keras metric function name. See                                  |
    |                                | strings      | `Keras documentation <https://keras.io/metrics/>`_.              |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | **training->model->optimizer**                                                                                   |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | type                           | string       | Keras optimizer name. See                                        |
    |                                |              | `Keras documentation <https://keras.io/optimizers/>`_.           |
    +--------------------------------+--------------+------------------------------------------------------------------+
    | parameters                     | dict         | Place inside this all parameters for the optimizer.              |
    +--------------------------------+--------------+------------------------------------------------------------------+

    """

    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'show_model_information': False,
            'audio_error_handling': False,
            'win_length_seconds': 0.1,
            'hop_length_seconds': 0.02,
            'method': 'mlp',
            'parameters': {
                'seed': 0,
                'keras': {
                    'backend': 'theano',
                    'backend_parameters': {
                        'CNR': True,
                        'device': 'cpu',
                        'fastmath': False,
                        'floatX': 'float64',
                        'openmp': False,
                        'optimizer': 'None',
                        'threads': 1
                    }
                },
                'model': {
                    'config': [
                        {
                            'class_name': 'Dense',
                            'config': {
                                'activation': 'relu',
                                'kernel_initializer': 'uniform',
                                'units': 50
                            }
                        },
                        {
                            'class_name': 'Dense',
                            'config': {
                                'activation': 'softmax',
                                'kernel_initializer': 'uniform',
                                'units': 'CLASS_COUNT'
                            }
                        }
                    ],
                    'loss': 'categorical_crossentropy',
                    'metrics': ['categorical_accuracy'],
                    'optimizer': {
                        'type': 'Adam'
                    }
                },
                'training': {
                    'batch_size': 256,
                    'epochs': 200,
                    'shuffle': True,
                    'callbacks': [],
                },
                'validation': {
                    'enable': True,
                    'setup_source': 'generated_scene_balanced',
                    'validation_amount': 0.1
                }
            }
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(SceneClassifierMLP, self).__init__(*args, **kwargs)
        self.method = 'mlp'

    def learn(self, data, annotations, data_filenames=None, validation_files=[], **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data
        validation_files: list of filenames
            Predefined validation files, use parameter 'validation.setup_source=dataset' to use them.

        Returns
        -------
        self

        """

        training_files = sorted(list(annotations.keys()))  # Collect training files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed')
                )

            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))

                else:
                    message = '{name}: No validation_files set'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            else:
                message = '{name}: Unknown validation.setup_source [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.learner_params.get_path('validation.setup_source')
                )

                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))

        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data=data, annotations=annotations)

        # Process data
        X_training = self.prepare_data(data=data, files=training_files)
        Y_training = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)

        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        # Process validation data
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))

        else:
            validation = None

        # Set seed
        self.set_seed()

        # Setup Keras
        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        # Create model
        self.create_model(input_shape=self._get_input_size(data=data))

        if self.show_extra_debug:
            self.log_model_summary()

        # Create callbacks
        callback_list = self.create_callback_list()

        if self.show_extra_debug:
            self.logger.debug('  Feature vector \t[{vector:d}]'.format(
                vector=self._get_input_size(data=data))
            )
            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )

            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )

        # Set seed
        self.set_seed()

        hist = self.model.fit(
            x=X_training,
            y=Y_training,
            batch_size=self.learner_params.get_path('training.batch_size', 1),
            epochs=self.learner_params.get_path('training.epochs', 1),
            validation_data=validation,
            verbose=0,
            shuffle=self.learner_params.get_path('training.shuffle', True),
            callbacks=callback_list
        )

        # Manually update callbacks
        for callback in callback_list:
            if hasattr(callback, 'close'):
                callback.close()

        for callback in callback_list:
            if isinstance(callback, StasherCallback):
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    self.model.set_weights(best_weights)
                break

        self['learning_history'] = hist.history

    def _frame_probabilities(self, feature_data):

        return self.model.predict(x=feature_data).T


class SceneClassifierKerasSequential(SceneClassifierMLP):
    """Sequential Keras model for Acoustic scene classification"""
    def __init__(self, *args, **kwargs):
        super(SceneClassifierKerasSequential, self).__init__(*args, **kwargs)
        self.method = 'keras_seq'

        self['data_processor'] = kwargs.get('data_processor')
        self['data_processor_training'] = kwargs.get('training_data_processor', copy.deepcopy(self['data_processor']))

        self.data_generators = kwargs.get('data_generators')
        if self.data_generators is None:
            self.data_generators = {}
            data_generator_list = get_class_inheritors(BaseDataGenerator)
            for data_generator_item in data_generator_list:
                generator = data_generator_item()
                if generator.method:
                    self.data_generators[generator.method] = data_generator_item

    @property
    def data_processor(self):
        """Feature processing chain

        Returns
        -------
         feature_processing_chain

        """

        return self.get('data_processor', None)

    @data_processor.setter
    def data_processor(self, value):
        self['data_processor'] = value

    @property
    def data_processor_training(self):
        """Feature processing chain

        Returns
        -------
         feature_processing_chain

        """

        return self.get('data_processor_training', None)

    @data_processor_training.setter
    def data_processor_training(self, value):
        self['data_processor_training'] = value

    def learn(self, data, annotations, data_filenames=None, validation_files=[], **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data
        validation_files: list of filenames
            Predefined validation files, use parameter 'validation.setup_source=dataset' to use them.

        Returns
        -------
        self

        """

        if (self.learner_params.get_path('temporal_shifting.enable') and
           not self.learner_params.get_path('generator.enable') and
           not self.learner_params.get_path('training.epoch_processing.enable')):

            message = '{name}: Temporal shifting cannot be used. Use epoch_processing or generator to allow temporal shifting of data.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed')
                )

            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))

                else:
                    message = '{name}: No validation_files set'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            else:
                message = '{name}: Unknown validation.setup_source [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.learner_params.get_path('validation.setup_source')
                )

                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))
        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Set seed
        self.set_seed()

        # Setup Keras
        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        if self.learner_params.get_path('generator.enable'):
            # Create generators
            if self.learner_params.get_path('generator.method') in self.data_generators:
                training_data_generator = self.data_generators[self.learner_params.get_path('generator.method')](
                    files=training_files,
                    data_filenames=data_filenames,
                    annotations=annotations,
                    data_processor=self.data_processor_training,
                    class_labels=self.class_labels,
                    hop_length_seconds=self.params.get_path('hop_length_seconds'),
                    shuffle=self.learner_params.get_path('training.shuffle', True),
                    batch_size=self.learner_params.get_path('training.batch_size', 1),
                    data_refresh_on_each_epoch=self.learner_params.get_path('temporal_shifting.enable'),
                    label_mode='scene',
                    **self.learner_params.get_path('generator.parameters', {})
                )

                if self.learner_params.get_path('validation.enable', False):
                    validation_data_generator = self.data_generators[self.learner_params.get_path('generator.method')](
                        files=validation_files,
                        data_filenames=data_filenames,
                        annotations=annotations,
                        data_processor=self.data_processor,
                        class_labels=self.class_labels,
                        hop_length_seconds=self.params.get_path('hop_length_seconds'),
                        shuffle=False,
                        batch_size=self.learner_params.get_path('training.batch_size', 1),
                        label_mode='scene',
                        **self.learner_params.get_path('generator.parameters', {})
                    )

                else:
                    validation_data_generator = None
            else:
                message = '{name}: Generator method not implemented [{method}]'.format(
                    name=self.__class__.__name__,
                    method=self.learner_params.get_path('generator.method')
                )
                self.logger.exception(message)
                raise ValueError(message)

            input_shape = training_data_generator.input_size
            training_data_size = training_data_generator.data_size
            if validation_data_generator:
                validation_data_size = validation_data_generator.data_size

        else:
            # Convert annotations into activity matrix format
            activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

            X_training = self.prepare_data(
                data=data,
                files=training_files,
                processor='training'
            )
            Y_training = self.prepare_activity(
                activity_matrix_dict=activity_matrix_dict,
                files=training_files,
                processor='training'
            )

            if validation_files:
                validation_data = (
                    self.prepare_data(
                        data=data,
                        files=validation_files,
                        processor='default'
                    ),
                    self.prepare_activity(
                        activity_matrix_dict=activity_matrix_dict,
                        files=validation_files,
                        processor='default'
                    )
                )
                validation_data_size = validation_data[0].shape[0]

            input_shape = X_training.shape[-1]
            training_data_size = X_training.shape[0]

        # Create model
        self.create_model(input_shape=input_shape)

        # Get processing interval
        processing_interval = self.get_processing_interval()

        # Create callbacks
        callback_list = self.create_callback_list()

        if self.show_extra_debug:
            self.log_model_summary()

            self.logger.debug('  Files')
            self.logger.debug(
                '    Training \t[{examples:d}]'.format(examples=training_data_size)
            )

            if validation_files:
                self.logger.debug(
                    '    Validation \t[{validation:d}]'.format(validation=validation_data_size)
                )
            self.logger.debug('  ')

            self.logger.debug('  Input')
            self.logger.debug('    Feature vector \t[{vector:d}]'.format(
                vector=input_shape)
            )

            if self.learner_params.get_path('input_sequencer.enable'):
                self.logger.debug('    Sequence\t[{length:d}]\t\t({time:4.2f} sec)'.format(
                    length=self.learner_params.get_path('input_sequencer.frames'),
                    time=self.learner_params.get_path('input_sequencer.frames')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('  ')

            if (self.learner_params.get_path('temporal_shifter.enable') and
               self.learner_params.get_path('training.epoch_processing.enable')):

                self.logger.debug('  Sequence shifting per epoch')
                self.logger.debug('    Shift \t\t[{step:d} per epoch]\t({time:4.2f} sec)'.format(
                    step=self.learner_params.get_path('temporal_shifter.step'),
                    time=self.learner_params.get_path('temporal_shifter.step')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('    Max \t\t[{max:d} per epoch]\t({time:4.2f} sec)'.format(
                    max=self.learner_params.get_path('temporal_shifter.max'),
                    time=self.learner_params.get_path('temporal_shifter.max')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('    Border \t\t[{border:s}]'.format(
                    border=self.learner_params.get_path('temporal_shifter.border', 'roll')
                ))
                self.logger.debug('  ')

            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )
            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )
            self.logger.debug('  ')

            # Extra info about training
            if self.learner_params.get_path('generator.enable'):
                if training_data_generator:
                    for i in training_data_generator.info():
                        self.logger.debug(i)

            if self.learner_params.get_path('training.epoch_processing.enable'):
                self.logger.debug('  Epoch processing \t[{mode}]'.format(
                    mode='Epoch-by-Epoch')
                )
            else:
                self.logger.debug('  Epoch processing \t[{mode}]'.format(
                    mode='Keras')
                )
            self.logger.debug('  ')

            if (self.learner_params.get_path('validation.enable') and
               self.learner_params.get_path('training.epoch_processing.enable') and
               self.learner_params.get_path('training.epoch_processing.external_metrics.enable')):

                self.logger.debug('  External metrics')

                self.logger.debug('    Metrics\t\tLabel\tEvaluator:Name')
                for metric in self.learner_params.get_path('training.epoch_processing.external_metrics.metrics'):
                    self.logger.debug('    \t\t[{label}]\t[{metric}]'.format(
                        label=metric.get('label'),
                        metric=metric.get('evaluator') + ':' + metric.get('name'))
                    )

                self.logger.debug('    Interval \t[{processing_interval:d} epochs]'.format(
                    processing_interval=processing_interval)
                )

            self.logger.debug('  ')

        # Set seed
        self.set_seed()

        epochs = self.learner_params.get_path('training.epochs', 1)

        # Initialize training history
        learning_history = {
            'loss': numpy.empty((epochs,)),
            'val_loss': numpy.empty((epochs,)),
        }
        for metric in self.model.metrics:
            learning_history[metric] = numpy.empty((epochs,))
            learning_history['val_'+metric] = numpy.empty((epochs,))
        for quantity in learning_history:
            learning_history[quantity][:] = numpy.nan

        if self.learner_params.get_path('training.epoch_processing.enable'):
            # Get external metric evaluators
            external_metric_evaluators = self.create_external_metric_evaluators()

            for external_metric_id in external_metric_evaluators:
                metric_label = external_metric_evaluators[external_metric_id]['label']
                learning_history[metric_label] = numpy.empty((epochs,))
                learning_history[metric_label][:] = numpy.nan

            for epoch_start in range(0, epochs, processing_interval):
                # Last epoch
                epoch_end = epoch_start + processing_interval
                # Make sure we have only specified amount of epochs
                if epoch_end > epochs:
                    epoch_end = epochs

                # Model fitting
                if self.learner_params.get_path('generator.enable'):
                    hist = self.model.fit_generator(
                        generator=training_data_generator.generator(),
                        steps_per_epoch=training_data_generator.steps_count,
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        validation_data=validation_data_generator.generator(),
                        validation_steps=validation_data_generator.steps_count,
                        max_queue_size=self.learner_params.get_path('generator.max_q_size', 1),
                        workers=self.learner_params.get_path('generator.workers', 1),
                        verbose=0,
                        callbacks=callback_list
                    )

                else:
                    hist = self.model.fit(
                        x=X_training,
                        y=Y_training,
                        batch_size=self.learner_params.get_path('training.batch_size', 1),
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        validation_data=validation_data,
                        verbose=0,
                        shuffle=self.learner_params.get_path('training.shuffle', True),
                        callbacks=callback_list
                    )

                # Store keras metrics into learning history log
                for keras_metric in hist.history:
                    learning_history[keras_metric][epoch_start:epoch_start+len(hist.history[keras_metric])] = hist.history[keras_metric]

                # Evaluate validation data with external metrics
                if (self.learner_params.get_path('validation.enable') and
                   self.learner_params.get_path('training.epoch_processing.external_metrics.enable')):

                    # Recognizer class
                    recognizer = SceneRecognizer(
                        params=self.learner_params.get_path('training.epoch_processing.recognizer'),
                        class_labels=self.class_labels
                    )

                    for external_metric_id in external_metric_evaluators:
                        # Reset evaluators
                        external_metric_evaluators[external_metric_id]['evaluator'].reset()

                        metric_label = external_metric_evaluators[external_metric_id]['label']

                        # Evaluate validation data
                        for validation_file in validation_files:
                            # Get feature data
                            if self.learner_params.get_path('generator.enable'):
                                feature_data, feature_data_length = self.data_processor.load(
                                    feature_filename_dict=data_filenames[validation_file]
                                )

                            else:
                                feature_data = data[validation_file]

                            # Predict
                            frame_probabilities = self.predict(feature_data=feature_data)

                            predicted = [
                                MetaDataItem(
                                    {
                                        'file': validation_file,
                                        'scene_label': recognizer.process(frame_probabilities=frame_probabilities),
                                    }
                                )
                            ]

                            # Get reference data
                            meta = [
                                annotations[validation_file]
                            ]

                            # Evaluate
                            external_metric_evaluators[external_metric_id]['evaluator'].evaluate(
                                meta,
                                predicted
                            )

                        # Get metric value
                        metric_value = DottedDict(
                            external_metric_evaluators[external_metric_id]['evaluator'].results()
                        ).get_path(external_metric_evaluators[external_metric_id]['path'])

                        if metric_value is None:
                            message = '{name}: Metric was not found, evaluator:[{evaluator}] metric:[{metric}]'.format(
                                name=self.__class__.__name__,
                                evaluator=external_metric_evaluators[external_metric_id]['evaluator'],
                                metric=external_metric_evaluators[external_metric_id]['path']
                            )
                            self.logger.exception(message)
                            raise ValueError(message)

                        # Inject external metric values to the callbacks
                        for callback in callback_list:
                            if hasattr(callback, 'set_external_metric_value'):
                                callback.set_external_metric_value(
                                    metric_label=metric_label,
                                    metric_value=metric_value
                                )

                        # Store metric value into learning history log
                        learning_history[metric_label][epoch_end-1] = metric_value

                # Manually update callbacks
                for callback in callback_list:
                    if hasattr(callback, 'update'):
                        callback.update()

                # Check we need to stop training
                stop_training = False
                for callback in callback_list:
                    if hasattr(callback, 'stop'):
                        if callback.stop():
                            stop_training = True

                if stop_training:
                    # Stop the training loop
                    break

                # Training data processing between epochs
                if self.learner_params.get_path('temporal_shifter.enable'):
                    # Increase temporal shifting
                    self.data_processor_training.call_method('increase_shifting')

                    if not self.learner_params.get_path('generator.enable'):
                        # Refresh training data manually with new parameters
                        X_training = self.prepare_data(
                            data=data,
                            files=training_files,
                            processor='training'
                        )

                        Y_training = self.prepare_activity(
                            activity_matrix_dict=activity_matrix_dict,
                            files=training_files,
                            processor='training'
                        )


        else:
            if self.learner_params.get_path('generator.enable'):
                hist = self.model.fit_generator(
                    generator=training_data_generator.generator(),
                    steps_per_epoch=training_data_generator.steps_count,
                    epochs=epochs,
                    validation_data=validation_data_generator.generator(),
                    validation_steps=validation_data_generator.steps_count,
                    max_queue_size=self.learner_params.get_path('generator.max_q_size', 1),
                    workers=1,
                    verbose=0,
                    callbacks=callback_list
                )

            else:
                hist = self.model.fit(
                    x=X_training,
                    y=Y_training,
                    batch_size=self.learner_params.get_path('training.batch_size', 1),
                    epochs=epochs,
                    validation_data=validation_data,
                    verbose=0,
                    shuffle=self.learner_params.get_path('training.shuffle', True),
                    callbacks=callback_list
                )

            # Store keras metrics into learning history log
            for keras_metric in hist.history:
                learning_history[keras_metric][0:len(hist.history[keras_metric])] = hist.history[keras_metric]

        # Manually update callbacks
        for callback in callback_list:
            if hasattr(callback, 'close'):
                callback.close()

        for callback in callback_list:
            if isinstance(callback, StasherCallback):
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    self.model.set_weights(best_weights)
                break

        # Store learning history to the model
        self['learning_history'] = learning_history

        self.logger.debug(' ')

    def predict(self, feature_data):
        """Predict frame probabilities for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Frame probabilities

        """

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        if isinstance(feature_data, dict) and self.data_processor:
            # Feature repository given, and feature processor present
            feature_data, feature_length = self.data_processor.process(feature_data=feature_data)

        # Get frame wise probabilities
        frame_probabilities = None
        if len(self.model.input_shape) == 2:
            frame_probabilities = self.model.predict(x=feature_data).T

        elif len(self.model.input_shape) == 4:
            if len(feature_data.shape) != 4:
                # Still feature data in wrong shape, trying to recover
                data_sequencer = DataSequencer(
                    frames=self.model.input_shape[2],
                    hop=self.model.input_shape[2],
                )

                feature_data = numpy.expand_dims(data_sequencer.process(feature_data), axis=1)

            frame_probabilities = self.model.predict(x=feature_data).T

            # Join sequences
            # TODO: if data_sequencer.hop != data_sequencer.frames, do additional processing here.
            frame_probabilities = frame_probabilities.reshape(
                frame_probabilities.shape[0],
                frame_probabilities.shape[1] * frame_probabilities.shape[2]
            )

        return frame_probabilities


class EventDetector(LearnerContainer):
    """Event detector (Frame classifier / Multi-class - Multi-label)"""

    def _get_target_matrix_dict(self, data, annotations):

        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            # Create event roll
            event_roll = EventRoll(metadata_container=annotations[audio_filename],
                                   label_list=self.class_labels,
                                   time_resolution=self.params.get_path('hop_length_seconds')
                                   )
            # Pad event roll to full length of the signal
            activity_matrix_dict[audio_filename] = event_roll.pad(length=data[audio_filename].shape[0])

        return activity_matrix_dict

    def _generate_validation(self, annotations, validation_type='generated_scene_location_event_balanced',
                             valid_percentage=0.20, seed=None):

        self.set_seed(seed=seed)
        validation_files = []

        if self.show_extra_debug:
            self.logger.debug('  Validation')

        if validation_type == 'generated_scene_location_event_balanced':
            # Get training data per scene label
            annotation_data = {}
            for audio_filename in sorted(list(annotations.keys())):
                scene_label = annotations[audio_filename][0].scene_label
                location_id = annotations[audio_filename][0].identifier
                if scene_label not in annotation_data:
                    annotation_data[scene_label] = {}
                if location_id not in annotation_data[scene_label]:
                    annotation_data[scene_label][location_id] = []
                annotation_data[scene_label][location_id].append(audio_filename)

            # Get event amounts
            event_amounts = {}
            for scene_label in list(annotation_data.keys()):
                if scene_label not in event_amounts:
                    event_amounts[scene_label] = {}
                for location_id in list(annotation_data[scene_label].keys()):
                    for audio_filename in annotation_data[scene_label][location_id]:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label, count in iteritems(current_event_amounts):
                            if event_label not in event_amounts[scene_label]:
                                event_amounts[scene_label][event_label] = 0
                            event_amounts[scene_label][event_label] += count

            for scene_label in list(annotation_data.keys()):
                # Optimize scene sets separately
                validation_set_candidates = []
                validation_set_MAE = []
                validation_set_event_amounts = []
                training_set_event_amounts = []
                for i in range(0, 1000):
                    location_ids = list(annotation_data[scene_label].keys())
                    random.shuffle(location_ids, random.random)

                    valid_percentage_index = int(numpy.ceil(valid_percentage * len(location_ids)))

                    current_validation_files = []
                    for loc_id in location_ids[0:valid_percentage_index]:
                        current_validation_files += annotation_data[scene_label][loc_id]

                    current_training_files = []
                    for loc_id in location_ids[valid_percentage_index:]:
                        current_training_files += annotation_data[scene_label][loc_id]

                    # event count in training set candidate
                    training_set_event_counts = numpy.zeros(len(event_amounts[scene_label]))
                    for audio_filename in current_training_files:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            if event_label in current_event_amounts:
                                training_set_event_counts[event_label_id] += current_event_amounts[event_label]

                    # Accept only sets which leave at least one example for training
                    if numpy.all(training_set_event_counts > 0):
                        # event counts in validation set candidate
                        validation_set_event_counts = numpy.zeros(len(event_amounts[scene_label]))
                        for audio_filename in current_validation_files:
                            current_event_amounts = annotations[audio_filename].event_stat_counts()

                            for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                                if event_label in current_event_amounts:
                                    validation_set_event_counts[event_label_id] += current_event_amounts[event_label]

                        # Accept only sets which have examples from each sound event class
                        if numpy.all(validation_set_event_counts > 0):
                            validation_amount = validation_set_event_counts / (validation_set_event_counts + training_set_event_counts)
                            validation_set_candidates.append(current_validation_files)
                            validation_set_MAE.append(mean_absolute_error(numpy.ones(len(validation_amount)) * valid_percentage, validation_amount))
                            validation_set_event_amounts.append(validation_set_event_counts)
                            training_set_event_amounts.append(training_set_event_counts)

                # Generate balance validation set
                # Selection done based on event counts (per scene class)
                # Target count specified percentage of training event count
                if validation_set_MAE:
                    best_set_id = numpy.argmin(validation_set_MAE)
                    validation_files += validation_set_candidates[best_set_id]

                    if self.show_extra_debug:
                        self.logger.debug('    Valid sets found [{sets}]'.format(
                            sets=len(validation_set_MAE))
                        )

                        self.logger.debug('    Best fitting set ID={id}, Error={error:4.2}%'.format(
                            id=best_set_id,
                            error=validation_set_MAE[best_set_id]*100)
                        )
                        self.logger.debug('    Validation event counts in respect of all data:')
                        event_amount_percentages = validation_set_event_amounts[best_set_id] / (validation_set_event_amounts[best_set_id] + training_set_event_amounts[best_set_id])
                        self.logger.debug('    {event:<20s} | {amount:10s} '.format(
                            event='Event label',
                            amount='Amount (%)')
                        )

                        self.logger.debug('    {event:<20s} + {amount:10s} '.format(
                            event='-' * 20,
                            amount='-' * 20)
                        )

                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            self.logger.debug('    {event:<20s} | {amount:4.2f} '.format(
                                event=event_label,
                                amount=numpy.round(event_amount_percentages[event_label_id] * 100))
                            )

                else:
                    message = '{name}: Validation setup creation was not successful! Could not find a set with ' \
                              'examples for each event class in both training and validation.'.format(
                                name=self.__class__.__name__
                              )

                    self.logger.exception(message)
                    raise AssertionError(message)

        elif validation_type == 'generated_event_file_balanced':
            # Get event amounts
            event_amounts = {}
            for audio_filename in sorted(list(annotations.keys())):
                event_label = annotations[audio_filename][0].event_label
                if event_label not in event_amounts:
                    event_amounts[event_label] = []
                event_amounts[event_label].append(audio_filename)

            if self.show_extra_debug:
                self.logger.debug('    {event_label:<20s} | {amount:20s} '.format(
                    event_label='Event label',
                    amount='Files (%)')
                )

                self.logger.debug('    {event_label:<20s} + {amount:20s} '.format(
                    event_label='-' * 20,
                    amount='-' * 20)
                )

            def sorter(key):
                if not key:
                    return ""
                return key

            event_label_list = list(event_amounts.keys())
            event_label_list.sort(key=sorter)
            for event_label in event_label_list:
                files = numpy.array(list(event_amounts[event_label]))
                random.shuffle(files, random.random)
                valid_percentage_index = int(numpy.ceil(valid_percentage * len(files)))
                validation_files += files[0:valid_percentage_index].tolist()

                if self.show_extra_debug:
                    self.logger.debug('    {event_label:<20s} | {amount:4.2f} '.format(
                        event_label=event_label if event_label else '-',
                        amount=valid_percentage_index / float(len(files)) * 100.0)
                    )

            random.shuffle(validation_files, random.random)

        else:
            message = '{name}: Unknown validation_type [{type}].'.format(
                name=self.__class__.__name__,
                type=validation_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

        if self.show_extra_debug:
            self.logger.debug(' ')

        return sorted(validation_files)

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        message = '{name}: Implement learn function.'.format(
            name=self.__class__.__name__
        )

        self.logger.exception(message)
        raise AssertionError(message)


class EventDetectorGMM(EventDetector):
    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'win_length_seconds': 0.04,
            'hop_length_seconds': 0.02,
            'method': 'gmm',
            'scene_handling': 'scene-dependent',
            'parameters': {
                'covariance_type': 'diag',
                'init_params': 'kmeans',
                'max_iter': 40,
                'n_components': 16,
                'n_init': 1,
                'random_state': 0,
                'reg_covar': 0,
                'tol': 0.001
            },
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(EventDetectorGMM, self).__init__(*args, **kwargs)
        self.method = 'gmm'

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data

        Returns
        -------
        self

        """

        from sklearn.mixture import GaussianMixture

        if not self.params.get_path('hop_length_seconds'):
            message = '{name}: No hop length set.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        if self.learner_params.get_path('validation.enable', False):
            message = '{name}: Validation is not implemented for this learner.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        class_progress = tqdm(
            self.class_labels,
            file=sys.stdout,
            leave=False,
            desc='           {0:>15s}'.format('Learn '),
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',  # [{elapsed}<{remaining}, {rate_fmt}]',
            disable=self.disable_progress_bar
        )

        # Collect training examples
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        for event_id, event_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {event_label:<15s}'.format(
                    title='Learn',
                    item_id=event_id,
                    total=len(self.class_labels),
                    event_label=event_label)
                )
            data_positive = []
            data_negative = []

            for audio_filename in sorted(list(activity_matrix_dict.keys())):
                activity_matrix = activity_matrix_dict[audio_filename]

                positive_mask = activity_matrix[:, event_id].astype(bool)
                # Store positive examples
                if any(positive_mask):
                    data_positive.append(data[audio_filename].feat[0][positive_mask, :])

                # Store negative examples
                if any(~positive_mask):
                    data_negative.append(data[audio_filename].feat[0][~positive_mask, :])

            self['model'][event_label] = {
                'positive': None,
                'negative': None,
            }

            if len(data_positive):
                self['model'][event_label]['positive'] = GaussianMixture(**self.learner_params).fit(
                    numpy.concatenate(data_positive)
                )

            if len(data_negative):
                self['model'][event_label]['negative'] = GaussianMixture(**self.learner_params).fit(
                    numpy.concatenate(data_negative)
                )

    def predict(self, feature_data):

        frame_probabilities_positive = numpy.empty((len(self.class_labels), feature_data.shape[0]))
        frame_probabilities_negative = numpy.empty((len(self.class_labels), feature_data.shape[0]))
        frame_probabilities_positive[:] = numpy.nan
        frame_probabilities_negative[:] = numpy.nan

        for event_id, event_label in enumerate(self.class_labels):
            if self['model'][event_label]['positive']:
                frame_probabilities_positive[event_id, :] = self['model'][event_label]['positive'].score_samples(
                    feature_data.feat[0]
                )

            if self['model'][event_label]['negative']:
                frame_probabilities_negative[event_id, :] = self['model'][event_label]['negative'].score_samples(
                    feature_data.feat[0]
                )

        return frame_probabilities_positive, frame_probabilities_negative


class EventDetectorGMMdeprecated(EventDetector):
    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'win_length_seconds': 0.04,
            'hop_length_seconds': 0.02,
            'method': 'gmm_deprecated',
            'scene_handling': 'scene-dependent',
            'parameters': {
                'n_components': 16,
                'covariance_type': 'diag',
                'random_state': 0,
                'tol': 0.001,
                'min_covar': 0.001,
                'n_iter': 40,
                'n_init': 1,
                'params': 'wmc',
                'init_params': 'wmc',
            },
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(EventDetectorGMMdeprecated, self).__init__(*args, **kwargs)
        self.method = 'gmm_deprecated'

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data

        Returns
        -------
        self

        """

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        if not self.params.get_path('hop_length_seconds'):
            message = '{name}: No hop length set.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        if self.learner_params.get_path('validation.enable', False):
            message = '{name}: Validation is not implemented for this learner.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        class_progress = tqdm(
            self.class_labels,
            file=sys.stdout,
            leave=False,
            desc='           {0:>15s}'.format('Learn '),
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',  # [{elapsed}<{remaining}, {rate_fmt}]',
            disable=self.disable_progress_bar
        )

        # Collect training examples
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        for event_id, event_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {event_label:<15s}'.format(
                    title='Learn',
                    item_id=event_id,
                    total=len(self.class_labels),
                    event_label=event_label)
                )
            data_positive = []
            data_negative = []

            for audio_filename in sorted(list(activity_matrix_dict.keys())):
                activity_matrix = activity_matrix_dict[audio_filename]

                positive_mask = activity_matrix[:, event_id].astype(bool)
                # Store positive examples
                if any(positive_mask):
                    data_positive.append(data[audio_filename].feat[0][positive_mask, :])

                # Store negative examples
                if any(~positive_mask):
                    data_negative.append(data[audio_filename].feat[0][~positive_mask, :])

            if event_label not in self['model']:
                self['model'][event_label] = {'positive': None, 'negative': None}

            self['model'][event_label] = {
                'positive': None,
                'negative': None,
            }
            if len(data_positive):
                self['model'][event_label]['positive'] = mixture.GMM(**self.learner_params).fit(
                    numpy.concatenate(data_positive)
                )

            if len(data_negative):
                self['model'][event_label]['negative'] = mixture.GMM(**self.learner_params).fit(
                    numpy.concatenate(data_negative)
                )

    def predict(self, feature_data):

        frame_probabilities_positive = numpy.empty((len(self.class_labels), feature_data.shape[0]))
        frame_probabilities_negative = numpy.empty((len(self.class_labels), feature_data.shape[0]))
        frame_probabilities_positive[:] = numpy.nan
        frame_probabilities_negative[:] = numpy.nan

        for event_id, event_label in enumerate(self.class_labels):
            if self['model'][event_label]['positive']:
                frame_probabilities_positive[event_id, :] = self['model'][event_label]['positive'].score_samples(
                    feature_data.feat[0]
                )[0]

            if self['model'][event_label]['negative']:
                frame_probabilities_negative[event_id, :] = self['model'][event_label]['negative'].score_samples(
                    feature_data.feat[0]
                )[0]

        return frame_probabilities_positive, frame_probabilities_negative


class EventDetectorMLP(EventDetector, KerasMixin):
    """Simple MLP based sequential Keras model for Sound Event Detection"""

    def __init__(self, *args, **kwargs):
        self.default_parameters = DottedDict({
            'win_length_seconds': 0.04,
            'hop_length_seconds': 0.02,
            'method': 'mlp',
            'scene_handling': 'scene-dependent',
            'parameters': {
                'seed': 0,
                'keras': {
                    'backend': 'theano',
                    'backend_parameters': {
                        'CNR': True,
                        'device': 'cpu',
                        'fastmath': False,
                        'floatX': 'float64',
                        'openmp': False,
                        'optimizer': 'None',
                        'threads': 1
                    }
                },
                'model': {
                    'config': [
                        {
                            'class_name': 'Dense',
                            'config': {
                                'activation': 'relu',
                                'kernel_initializer': 'uniform',
                                'units': 50
                            }
                        },
                        {
                            'class_name': 'Dense',
                            'config': {
                                'activation': 'sigmoid',
                                'kernel_initializer': 'uniform',
                                'units': 'CLASS_COUNT'
                            }
                        }
                    ],
                    'loss': 'categorical_crossentropy',
                    'metrics': ['categorical_accuracy'],
                    'optimizer': {
                        'type': 'Adam'
                    }
                },
                'training': {
                    'batch_size': 256,
                    'epochs': 200,
                    'shuffle': True,
                    'callbacks': [],
                },
                'validation': {
                    'enable': True,
                    'setup_source': 'generated_scene_location_event_balanced',
                    'validation_amount': 0.1
                }
            },
        })
        self.default_parameters.merge(override=kwargs.get('params', {}))
        kwargs['params'] = self.default_parameters

        super(EventDetectorMLP, self).__init__(*args, **kwargs)
        self.method = 'mlp'

    def learn(self, data, annotations, data_filenames=None, validation_files=[], **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data
        validation_files: list of filenames
            Predefined validation files, use parameter 'validation.setup_source=dataset' to use them.

        -------
        self

        """

        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed'),
                )

            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))

                else:
                    message = '{name}: No validation_files set'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            else:
                message = '{name}: Unknown validation.setup_source [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.learner_params.get_path('validation.setup_source')
                )

                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))

        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        # Process data
        X_training = self.prepare_data(data=data, files=training_files)
        Y_training = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)

        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        # Process validation data
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

            validation = (X_validation, Y_validation)

            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))
        else:
            validation = None

        # Set seed
        self.set_seed()

        # Setup Keras
        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        # Create model
        self.create_model(input_shape=self._get_input_size(data=data))

        if self.show_extra_debug:
            self.log_model_summary()

        class_weight = None
        if len(self.class_labels) == 1:
            # Special case with binary classifier
            if self.learner_params.get_path('training.class_weight'):
                class_weight = {}
                for class_id, weight in enumerate(self.learner_params.get_path('training.class_weight')):
                    class_weight[class_id] = float(weight)

            if self.show_extra_debug:
                negative_examples_id = numpy.where(Y_training[:, 0] == 0)[0]
                positive_examples_id = numpy.where(Y_training[:, 0] == 1)[0]

                self.logger.debug('  Positives items \t[{positives:d}]\t({percentage:.2f} %)'.format(
                    positives=len(positive_examples_id),
                    percentage=len(positive_examples_id)/float(len(positive_examples_id)+len(negative_examples_id))*100
                ))
                self.logger.debug('  Negatives items \t[{negatives:d}]\t({percentage:.2f} %)'.format(
                    negatives=len(negative_examples_id),
                    percentage=len(negative_examples_id) / float(len(positive_examples_id) + len(negative_examples_id)) * 100
                ))

                self.logger.debug('  Class weights \t[{weights}]\t'.format(weights=class_weight))

        callback_list = self.create_callback_list()

        if self.show_extra_debug:
            self.logger.debug('  Feature vector \t[{vector:d}]'.format(
                vector=self._get_input_size(data=data))
            )
            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )
            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )

        # Set seed
        self.set_seed()

        hist = self.model.fit(
            x=X_training,
            y=Y_training,
            batch_size=self.learner_params.get_path('training.batch_size', 1),
            epochs=self.learner_params.get_path('training.epochs', 1),
            validation_data=validation,
            verbose=0,
            shuffle=self.learner_params.get_path('training.shuffle', True),
            callbacks=callback_list,
            class_weight=class_weight
        )

        # Manually update callbacks
        for callback in callback_list:
            if hasattr(callback, 'close'):
                callback.close()

        for callback in callback_list:
            if isinstance(callback, StasherCallback):
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    self.model.set_weights(best_weights)
                break

        self['learning_history'] = hist.history

    def predict(self, feature_data):

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]

        return self.model.predict(x=feature_data).T


class EventDetectorKerasSequential(EventDetectorMLP):
    """Sequential Keras model for Sound Event Detection"""

    def __init__(self, *args, **kwargs):
        super(EventDetectorKerasSequential, self).__init__(*args, **kwargs)
        self.method = 'keras_seq'

        self['data_processor'] = kwargs.get('data_processor')
        self['data_processor_training'] = kwargs.get('training_data_processor', copy.deepcopy(self['data_processor']))

        self.data_generators = kwargs.get('data_generators')
        if self.data_generators is None:
            self.data_generators = {}
            data_generator_list = get_class_inheritors(BaseDataGenerator)
            for data_generator_item in data_generator_list:
                generator = data_generator_item()
                if generator.method:
                    self.data_generators[generator.method] = data_generator_item

    @property
    def data_processor(self):
        """Feature processing chain

        Returns
        -------
         feature_processing_chain

        """

        return self.get('data_processor', None)

    @data_processor.setter
    def data_processor(self, value):
        self['data_processor'] = value

    @property
    def data_processor_training(self):
        """Feature processing chain

        Returns
        -------
         feature_processing_chain

        """

        return self.get('data_processor_training', None)

    @data_processor_training.setter
    def data_processor_training(self, value):
        self['data_processor_training'] = value

    def learn(self, annotations, data=None, data_filenames=None, validation_files=[], **kwargs):
        """Learn based on data and annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data
        data_filenames : dict of filenames
            Filenames of stored data
        validation_files: list of filenames
            Predefined validation files, use parameter 'validation.setup_source=dataset' to use them.

        Returns
        -------
        self

        """

        if (self.learner_params.get_path('temporal_shifting.enable') and
           not self.learner_params.get_path('generator.enable') and
           not self.learner_params.get_path('training.epoch_processing.enable')):

            message = '{name}: Temporal shifting cannot be used. Use epoch_processing or generator to allow temporal ' \
                      'shifting of data.'.format(
                        name=self.__class__.__name__
                      )

            self.logger.exception(message)
            raise ValueError(message)

        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed')
                )

            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))

                else:
                    message = '{name}: No validation_files set'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            else:
                message = '{name}: Unknown validation.setup_source [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.learner_params.get_path('validation.setup_source')
                )

                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))

        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Set seed
        self.set_seed()

        # Setup Keras
        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        if self.learner_params.get_path('generator.enable'):
            # Create generators
            if self.learner_params.get_path('generator.method') in self.data_generators:
                training_data_generator = self.data_generators[self.learner_params.get_path('generator.method')](
                    files=training_files,
                    data_filenames=data_filenames,
                    annotations=annotations,
                    data_processor=self.data_processor_training,
                    class_labels=self.class_labels,
                    hop_length_seconds=self.params.get_path('hop_length_seconds'),
                    shuffle=self.learner_params.get_path('training.shuffle', True),
                    batch_size=self.learner_params.get_path('training.batch_size', 1),
                    data_refresh_on_each_epoch=self.learner_params.get_path('temporal_shifting.enable'),
                    label_mode='event',
                    **self.learner_params.get_path('generator.parameters', {})
                )

                if self.learner_params.get_path('validation.enable', False):
                    validation_data_generator = self.data_generators[self.learner_params.get_path('generator.method')](
                        files=validation_files,
                        data_filenames=data_filenames,
                        annotations=annotations,
                        data_processor=self.data_processor,
                        class_labels=self.class_labels,
                        hop_length_seconds=self.params.get_path('hop_length_seconds'),
                        shuffle=False,
                        batch_size=self.learner_params.get_path('training.batch_size', 1),
                        label_mode='event',
                        **self.learner_params.get_path('generator.parameters', {})
                    )

                else:
                    validation_data_generator = None

            else:
                message = '{name}: Generator method not implemented [{method}]'.format(
                    name=self.__class__.__name__,
                    method=self.learner_params.get_path('generator.method')
                )
                self.logger.exception(message)
                raise ValueError(message)

            input_shape = training_data_generator.input_size
            training_data_size = training_data_generator.data_size
            if validation_data_generator:
                validation_data_size = validation_data_generator.data_size

        else:
            # Convert annotations into activity matrix format
            activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

            X_training = self.prepare_data(
                data=data,
                files=training_files,
                processor='training'
            )

            Y_training = self.prepare_activity(
                activity_matrix_dict=activity_matrix_dict,
                files=training_files,
                processor='training'
            )

            if validation_files:
                validation_data = (
                    self.prepare_data(
                        data=data,
                        files=validation_files,
                        processor='default'
                    ),
                    self.prepare_activity(
                        activity_matrix_dict=activity_matrix_dict,
                        files=validation_files,
                        processor='default'
                    )
                )

                validation_data_size = validation_data[0].shape[0]

            input_shape = X_training.shape[-1]
            training_data_size = X_training.shape[0]

        # Create model
        self.create_model(input_shape=input_shape)

        # Get processing interval
        processing_interval = self.get_processing_interval()

        # Create callbacks
        callback_list = self.create_callback_list()

        if self.show_extra_debug:
            self.log_model_summary()

            self.logger.debug('  Files')
            self.logger.debug(
                '    Training \t[{examples:d}]'.format(examples=training_data_size)
            )

            if validation_files:
                self.logger.debug(
                    '    Validation \t[{validation:d}]'.format(validation=validation_data_size)
                )
            self.logger.debug('  ')

        class_weight = None
        if len(self.class_labels) == 1:
            # Special case with binary classifier
            if self.learner_params.get_path('training.class_weight'):
                class_weight = {}
                for class_id, weight in enumerate(self.learner_params.get_path('training.class_weight')):
                    class_weight[class_id] = float(weight)

            if self.show_extra_debug and not self.learner_params.get_path('generator.enable'):
                if len(Y_training.shape) == 2:
                    negative_examples_id = numpy.where(Y_training[:, 0] == 0)[0]
                    positive_examples_id = numpy.where(Y_training[:, 0] == 1)[0]

                elif len(Y_training.shape) == 3:
                    negative_examples_id = numpy.where(Y_training[:, :, 0] == 0)[0]
                    positive_examples_id = numpy.where(Y_training[:, :, 0] == 1)[0]

                self.logger.debug('  Items')
                self.logger.debug('    Positive \t[{perc:.2f} %]\t({positives:d})'.format(
                    positives=len(positive_examples_id),
                    perc=len(positive_examples_id)/float(len(positive_examples_id)+len(negative_examples_id))*100
                ))
                self.logger.debug('    Negative \t[{perc:.2f} %]\t({negatives:d})'.format(
                    negatives=len(negative_examples_id),
                    perc=len(negative_examples_id) / float(len(positive_examples_id) + len(negative_examples_id)) * 100
                ))
                self.logger.debug('  Class weights \t[{weights}]\t'.format(weights=class_weight))
                self.logger.debug('  ')

        if self.show_extra_debug:
            self.logger.debug('  Input')
            self.logger.debug('    Feature vector \t[{vector:d}]'.format(
                vector=input_shape)
            )

            if self.learner_params.get_path('input_sequencer.enable'):
                self.logger.debug('    Sequence\t[{length:d}]\t\t({time:4.2f} sec)'.format(
                    length=self.learner_params.get_path('input_sequencer.frames'),
                    time=self.learner_params.get_path('input_sequencer.frames')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('  ')

            if (self.learner_params.get_path('temporal_shifter.enable') and
               self.learner_params.get_path('training.epoch_processing.enable')):

                self.logger.debug('  Sequence shifting per epoch')
                self.logger.debug('    Shift \t\t[{step:d} per epoch]\t({time:4.2f} sec)'.format(
                    step=self.learner_params.get_path('temporal_shifter.step'),
                    time=self.learner_params.get_path('temporal_shifter.step')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('    Max \t\t[{max:d} per epoch]\t({time:4.2f} sec)'.format(
                    max=self.learner_params.get_path('temporal_shifter.max'),
                    time=self.learner_params.get_path('temporal_shifter.max')*self.params.get_path('hop_length_seconds')
                    )
                )
                self.logger.debug('    Border \t\t[{border:s}]'.format(
                    border=self.learner_params.get_path('temporal_shifter.border', 'roll')
                ))
                self.logger.debug('  ')

            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )
            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )
            self.logger.debug('  ')

            # Extra info about training
            if self.learner_params.get_path('generator.enable'):
                if training_data_generator:
                    for i in training_data_generator.info():
                        self.logger.debug(i)

            if self.learner_params.get_path('training.epoch_processing.enable'):
                self.logger.debug('  Epoch processing \t[{mode}]'.format(
                    mode='Epoch-by-Epoch')
                )
            else:
                self.logger.debug('  Epoch processing \t[{mode}]'.format(
                    mode='Keras')
                )
            self.logger.debug('  ')

            if (self.learner_params.get_path('validation.enable') and
               self.learner_params.get_path('training.epoch_processing.enable') and
               self.learner_params.get_path('training.epoch_processing.external_metrics.enable')):

                self.logger.debug('  External metrics')

                self.logger.debug('    Metrics\t\tLabel\tEvaluator:Name')
                for metric in self.learner_params.get_path('training.epoch_processing.external_metrics.metrics'):
                    self.logger.debug('    \t\t[{label}]\t[{metric}]'.format(
                        label=metric.get('label'),
                        metric=metric.get('evaluator') + ':' + metric.get('name'))
                    )

                self.logger.debug('    Interval \t[{processing_interval:d} epochs]'.format(
                    processing_interval=processing_interval)
                )

            self.logger.debug('  ')

        # Set seed
        self.set_seed()

        epochs = self.learner_params.get_path('training.epochs', 1)

        # Initialize training history
        learning_history = {
            'loss': numpy.empty((epochs,)),
            'val_loss': numpy.empty((epochs,)),
        }
        for metric in self.model.metrics:
            learning_history[metric] = numpy.empty((epochs,))
            learning_history['val_'+metric] = numpy.empty((epochs,))
        for quantity in learning_history:
            learning_history[quantity][:] = numpy.nan

        if self.learner_params.get_path('training.epoch_processing.enable'):
            # Get external metric evaluators
            external_metric_evaluators = self.create_external_metric_evaluators()

            for external_metric_id in external_metric_evaluators:
                metric_label = external_metric_evaluators[external_metric_id]['label']
                learning_history[metric_label] = numpy.empty((epochs,))
                learning_history[metric_label][:] = numpy.nan

            for epoch_start in range(0, epochs, processing_interval):
                # Last epoch
                epoch_end = epoch_start + processing_interval
                # Make sure we have only specified amount of epochs
                if epoch_end > epochs:
                    epoch_end = epochs

                # Model fitting
                if self.learner_params.get_path('generator.enable'):
                    hist = self.model.fit_generator(
                        generator=training_data_generator.generator(),
                        steps_per_epoch=training_data_generator.steps_count,
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        validation_data=validation_data_generator.generator(),
                        validation_steps=validation_data_generator.steps_count,
                        max_queue_size=self.learner_params.get_path('generator.max_q_size', 1),
                        workers=self.learner_params.get_path('generator.workers', 1),
                        verbose=0,
                        callbacks=callback_list,
                        class_weight=class_weight
                    )

                else:
                    hist = self.model.fit(
                        x=X_training,
                        y=Y_training,
                        batch_size=self.learner_params.get_path('training.batch_size', 1),
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        validation_data=validation_data,
                        verbose=0,
                        shuffle=self.learner_params.get_path('training.shuffle', True),
                        callbacks=callback_list,
                        class_weight=class_weight
                    )

                # Store keras metrics into learning history log
                for keras_metric in hist.history:
                    learning_history[keras_metric][epoch_start:epoch_start+len(hist.history[keras_metric])] = hist.history[keras_metric]

                # Evaluate validation data with external metrics
                if (self.learner_params.get_path('validation.enable') and
                   self.learner_params.get_path('training.epoch_processing.external_metrics.enable')):

                    # Recognizer class
                    recognizer = EventRecognizer(
                        hop_length_seconds=self.params.get_path('hop_length_seconds'),
                        params=self.learner_params.get_path('training.epoch_processing.recognizer'),
                        class_labels=self.class_labels
                    )

                    for external_metric_id in external_metric_evaluators:
                        # Reset evaluators
                        external_metric_evaluators[external_metric_id]['evaluator'].reset()

                        metric_label = external_metric_evaluators[external_metric_id]['label']

                        # Evaluate validation data
                        for validation_file in validation_files:
                            # Get feature data
                            if self.learner_params.get_path('generator.enable'):
                                feature_data, feature_data_length = self.data_processor.load(
                                    feature_filename_dict=data_filenames[validation_file]
                                )

                            else:
                                feature_data = data[validation_file]

                            frame_probabilities = self.predict(feature_data=feature_data)

                            # Predict
                            predicted = recognizer.process(frame_probabilities=frame_probabilities)

                            # Get reference data
                            meta = []
                            for meta_item in annotations[validation_file]:
                                if 'event_label' in meta_item and meta_item.event_label:
                                    meta.append(meta_item)

                            # Evaluate
                            external_metric_evaluators[external_metric_id]['evaluator'].evaluate(
                                reference_event_list=meta,
                                estimated_event_list=predicted
                            )

                        # Get metric value
                        metric_value = DottedDict(
                            external_metric_evaluators[external_metric_id]['evaluator'].results()
                        ).get_path(external_metric_evaluators[external_metric_id]['path'])

                        if metric_value is None:
                            message = '{name}: Metric was not found, evaluator:[{evaluator}] metric:[{metric}]'.format(
                                name=self.__class__.__name__,
                                evaluator=external_metric_evaluators[external_metric_id]['evaluator'],
                                metric=external_metric_evaluators[external_metric_id]['path']
                            )
                            self.logger.exception(message)
                            raise ValueError(message)

                        # Inject external metric values to the callbacks
                        for callback in callback_list:
                            if hasattr(callback, 'set_external_metric_value'):
                                callback.set_external_metric_value(
                                    metric_label=metric_label,
                                    metric_value=metric_value
                                )

                        # Store metric value into learning history log
                        learning_history[metric_label][epoch_end-1] = metric_value

                # Manually update callbacks
                for callback in callback_list:
                    if hasattr(callback, 'update'):
                        callback.update()

                # Check we need to stop training
                stop_training = False
                for callback in callback_list:
                    if hasattr(callback, 'stop'):
                        if callback.stop():
                            stop_training = True

                if stop_training:
                    # Stop the training loop
                    break

                # Training data processing between epochs
                if self.learner_params.get_path('temporal_shifter.enable'):
                    # Increase temporal shifting
                    self.data_processor_training.call_method('increase_shifting')

                    if not self.learner_params.get_path('generator.enable'):
                        # Refresh training data manually with new parameters
                        X_training = self.prepare_data(
                            data=data,
                            files=training_files,
                            processor='training'
                        )
                        Y_training = self.prepare_activity(
                            activity_matrix_dict=activity_matrix_dict,
                            files=training_files,
                            processor='training'
                        )

        else:
            if self.learner_params.get_path('generator.enable'):
                hist = self.model.fit_generator(
                    generator=training_data_generator.generator(),
                    steps_per_epoch=training_data_generator.steps_count,
                    epochs=epochs,
                    validation_data=validation_data_generator.generator(),
                    validation_steps=validation_data_generator.steps_count,
                    max_queue_size=self.learner_params.get_path('generator.max_q_size', 1),
                    workers=1,
                    verbose=0,
                    callbacks=callback_list,
                    class_weight=class_weight
                )

            else:
                hist = self.model.fit(
                    x=X_training,
                    y=Y_training,
                    batch_size=self.learner_params.get_path('training.batch_size', 1),
                    epochs=epochs,
                    validation_data=validation_data,
                    verbose=0,
                    shuffle=self.learner_params.get_path('training.shuffle', True),
                    callbacks=callback_list,
                    class_weight=class_weight
                )

            # Store keras metrics into learning history log
            for keras_metric in hist.history:
                learning_history[keras_metric][0:len(hist.history[keras_metric])] = hist.history[keras_metric]

        # Manually update callbacks
        for callback in callback_list:
            if hasattr(callback, 'close'):
                callback.close()

        for callback in callback_list:
            if isinstance(callback, StasherCallback):
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    self.model.set_weights(best_weights)
                break

        # Store learning history to the model
        self['learning_history'] = learning_history

        self.logger.debug(' ')

    def predict(self, feature_data):

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]

        if isinstance(feature_data, dict) and self.data_processor:
            # Feature repository given, and feature processor present
            feature_data, feature_length = self.data_processor.process(feature_data=feature_data)

        # Frame probabilities
        frame_probabilities = None
        if len(self.model.input_shape) == 2:
            frame_probabilities = self.model.predict(x=feature_data).T

        elif len(self.model.input_shape) == 4:
            if len(feature_data.shape) != 4:
                # Still feature data in wrong shape, trying to recover
                data_sequencer = DataSequencer(
                    frames=self.model.input_shape[2],
                    hop=self.model.input_shape[2],
                )
                feature_data = numpy.expand_dims(data_sequencer.process(feature_data), axis=1)

            frame_probabilities = self.model.predict(x=feature_data).T

            # Join sequences
            # TODO: if data_sequencer.hop != data_sequencer.frames, do additional processing here.
            frame_probabilities = frame_probabilities.reshape(
                frame_probabilities.shape[0],
                frame_probabilities.shape[1] * frame_probabilities.shape[2]
            )

        return frame_probabilities

