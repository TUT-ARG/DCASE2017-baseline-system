#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learners
==================
Classes for machine learning

SceneClassifier
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    SceneClassifier

    SceneClassifierGMM
    SceneClassifierGMM.learn
    SceneClassifierGMM.predict

    SceneClassifierMLP
    SceneClassifierMLP.learn
    SceneClassifierMLP.predict

EventDetector
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    EventDetector

    EventDetectorGMM
    EventDetectorGMM.learn
    EventDetectorGMM.predict

    EventDetectorMLP
    EventDetectorMLP.learn
    EventDetectorMLP.predict


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
import os
import numpy
import logging
import random
import warnings
import importlib
import copy
import scipy

from sklearn.metrics import mean_absolute_error

from datetime import datetime
from .files import DataFile
from .containers import ContainerMixin, DottedDict
from .features import FeatureContainer
from .utils import SuppressStdoutAndStderr
from .metadata import MetaDataContainer, MetaDataItem, EventRoll
from .utils import Timer

from tqdm import tqdm


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
            'feature_processing_chain': kwargs.get('feature_processing_chain', None),
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
    def feature_processing_chain(self):
        """Feature processing chain

        Returns
        -------
         feature_processing_chain

        """

        return self.get('feature_processing_chain', None)

    @feature_processing_chain.setter
    def feature_processing_chain(self, value):
        self['feature_processing_chain'] = value

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


class KerasMixin(object):

    def keras_model_exists(self):
        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return os.path.isfile(self.filename) and os.path.isfile(keras_model_filename)

    def log_model_summary(self):
        layer_name_map = {
            'BatchNormalization': 'BatchNorm',
        }
        import keras.backend as K

        self.logger.debug('  ')
        self.logger.debug('  Model summary')
        self.logger.debug(
            '  {type:<12s} | {out:10s} | {param:6s}  | {name:20s}  | {conn:27s} | {act:10s} | {init:9s}'.format(
                type='Layer type',
                out='Output',
                param='Param',
                name='Name',
                conn='Connected to',
                act='Activation',
                init='Init')
        )

        self.logger.debug(
            '  {type:<12s} + {out:10s} + {param:6s}  + {name:20s}  + {conn:27s} + {act:10s} + {init:9s}'.format(
                type='-'*12,
                out='-'*10,
                param='-'*6,
                name='-'*20,
                conn='-'*27,
                act='-'*10,
                init='-'*9)
        )

        for layer in self.model.layers:
            connections = []
            for node_index, node in enumerate(layer.inbound_nodes):
                for i in range(len(node.inbound_layers)):
                    inbound_layer = node.inbound_layers[i].name
                    inbound_node_index = node.node_indices[i]
                    inbound_tensor_index = node.tensor_indices[i]
                    connections.append(inbound_layer + '[' + str(inbound_node_index) +
                                       '][' + str(inbound_tensor_index) + ']')

            config = layer.get_config()
            layer_name = layer.__class__.__name__
            if layer_name in layer_name_map:
                layer_name = layer_name_map[layer_name]

            self.logger.debug(
                '  {type:<12s} | {shape:10s} | {params:6s}  | {name:20s}  | {connected:27s} | {activation:10s} | {init:9s}'.format(
                    type=layer_name,
                    shape=str(layer.output_shape),
                    params=str(layer.count_params()),
                    name=str(layer.name),
                    connected=str(connections[0]) if len(connections)>0 else '---',
                    activation=str(config.get('activation', '---')),
                    init=str(config.get('init', '---'))
                )
            )
        trainable_count = int(numpy.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        non_trainable_count = int(numpy.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))

        self.logger.debug('  Total params         : {param_count:,}'.format(param_count=int(trainable_count + non_trainable_count)))
        self.logger.debug('  Trainable params     : {param_count:,}'.format(param_count=int(trainable_count)))
        self.logger.debug('  Non-trainable params : {param_count:,}'.format(param_count=int(non_trainable_count)))
        self.logger.debug('  ')
        
    def plot_model(self, filename='model.png', show_shapes=True, show_layer_names=True):
        from keras.utils.visualize_util import plot
        plot(self.model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)

    def process_data(self, data, files):
        """Concatenate feature data into one feature matrix

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        files : list of str
            List of filenames
        Returns
        -------
        numpy.ndarray
            Features concatenated
        """

        return numpy.vstack([data[x].feat[0] for x in files])

    def process_activity(self, activity_matrix_dict, files):
        """Concatenate activity matrices into one activity matrix

        Parameters
        ----------
        activity_matrix_dict : dict of binary matrices
            Meta data
        files : list of str
            List of filenames
        Returns
        -------
        numpy.ndarray
            Activity matrix
        """

        return numpy.vstack([activity_matrix_dict[x] for x in files])

    def create_model(self, input_shape):
        from keras.models import Sequential
        self.model = Sequential()

        # Get model parameters
        model_params = copy.deepcopy(self.learner_params.get_path('model.config'))

        # Setup layers
        for layer_id, layer_setup in enumerate(model_params):
            # Get layer parameters
            layer_setup = DottedDict(layer_setup)
            if 'config' not in layer_setup:
                layer_setup['config'] = {}

            # Get layer class
            try:
                LayerClass = getattr(
                    importlib.import_module("keras.layers"),
                    layer_setup['class_name']
                )

            except AttributeError:
                message = '{name}: Invalid Keras layer type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=layer_setup['class_name']
                )
                self.logger.exception(message)
                raise AttributeError(message)

            # Convert input_shape into tuple if list is given
            if 'input_shape' in layer_setup['config'] and isinstance(layer_setup['config']['input_shape'], list):
                if 'FEATURE_VECTOR_LENGTH' in layer_setup['config']['input_shape']:
                    layer_setup['config']['input_shape'][layer_setup['config']['input_shape'].index('FEATURE_VECTOR_LENGTH')] = input_shape
                elif 'CLASS_COUNT' in layer_setup['config']['input_shape']:
                    layer_setup['config']['input_shape'][layer_setup['config']['input_shape'].index('CLASS_COUNT')] = len(self.class_labels)

                layer_setup['config']['input_shape'] = tuple(layer_setup['config']['input_shape'])

            # Layer setup
            if layer_id == 0 and layer_setup.get_path('config.input_shape') is None:
                # Set input layer dimension for the first layer if not set
                layer_setup['config']['input_shape'] = (input_shape,)

            elif layer_setup.get_path('config.input_dim') == 'FEATURE_VECTOR_LENGTH':
                # Magic field "FEATURE_VECTOR_LENGTH"
                layer_setup['config']['input_shape'] = (input_shape,)

            elif layer_setup.get_path('config.input_shape') == 'FEATURE_VECTOR_LENGTH':
                # Magic field "FEATURE_VECTOR_LENGTH"
                layer_setup['config']['input_shape'] = (input_shape,)

            # Set layer output
            if layer_setup.get_path('config.units') == 'CLASS_COUNT':
                # Magic field "CLASS_COUNT"
                layer_setup['config']['units'] = len(self.class_labels)

            if layer_setup.get('config'):
                self.model.add(LayerClass(**dict(layer_setup.get('config'))))
            else:
                self.model.add(LayerClass())

        # Get Optimizer class
        try:
            OptimizerClass = getattr(
                importlib.import_module("keras.optimizers"),
                self.learner_params.get_path('model.optimizer.type')
            )

        except AttributeError:
            message = '{name}: Invalid Keras optimizer type [{type}].'.format(
                name=self.__class__.__name__,
                type=self.learner_params.get_path('model.optimizer.type')
            )
            self.logger.exception(message)
            raise AttributeError(message)

        # Compile the model
        self.model.compile(
            loss=self.learner_params.get_path('model.loss'),
            optimizer=OptimizerClass(**dict(self.learner_params.get_path('model.optimizer.parameters', {}))),
            metrics=self.learner_params.get_path('model.metrics')
        )

    def __getstate__(self):
        data = {}
        excluded_fields = ['model']

        for item in self:
            if item not in excluded_fields and self.get(item):
                data[item] = copy.deepcopy(self.get(item))
        data['model'] = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return data

    def _after_load(self, to_return=None):
        with SuppressStdoutAndStderr():
            from keras.models import Sequential, load_model

        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'

        if os.path.isfile(keras_model_filename):
            with SuppressStdoutAndStderr():
                self.model = load_model(keras_model_filename)
        else:
            message = '{name}: Keras model not found [{filename}]'.format(
                name=self.__class__.__name__,
                filename=keras_model_filename
            )

            self.logger.exception(message)
            raise IOError(message)

    def _after_save(self, to_return=None):
        # Save keras model and weight
        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'
        model_weights_filename = os.path.splitext(self.filename)[0] + '.weights.hdf5'
        self.model.save(keras_model_filename)
        self.model.save_weights(model_weights_filename)

    def _setup_keras(self):
        """Setup keras backend and parameters"""

        # Get BLAS library associated to numpy
        if numpy.__config__.blas_opt_info and 'libraries' in numpy.__config__.blas_opt_info:
            blas_libraries = numpy.__config__.blas_opt_info['libraries']
        else:
            blas_libraries = ['']

        blas_extra_info = []
        # Set backend and parameters before importing keras
        if self.show_extra_debug:
            self.logger.debug('  ')
            self.logger.debug('  Keras backend \t[{backend}]'.format(
                backend=self.learner_params.get_path('keras.backend', 'theano'))
            )

        # Threading
        if self.learner_params.get_path('keras.backend_parameters.threads'):
            thread_count = self.learner_params.get_path('keras.backend_parameters.threads', 1)
            os.environ['GOTO_NUM_THREADS'] = str(thread_count)
            os.environ['OMP_NUM_THREADS'] = str(thread_count)
            os.environ['MKL_NUM_THREADS'] = str(thread_count)
            blas_extra_info.append('Threads[{threads}]'.format(threads=thread_count))

            if thread_count > 1:
                os.environ['OMP_DYNAMIC'] = 'False'
                os.environ['MKL_DYNAMIC'] = 'False'
            else:
                os.environ['OMP_DYNAMIC'] = 'True'
                os.environ['MKL_DYNAMIC'] = 'True'

        # Conditional Numerical Reproducibility (CNR) for MKL BLAS library
        if self.learner_params.get_path('keras.backend_parameters.CNR', True) and blas_libraries[0].startswith('mkl'):
            os.environ['MKL_CBWR'] = 'COMPATIBLE'
            blas_extra_info.append('MKL_CBWR[{mode}]'.format(mode='COMPATIBLE'))

        # Show BLAS info
        if self.show_extra_debug:
            if numpy.__config__.blas_opt_info:
                blas_libraries = numpy.__config__.blas_opt_info['libraries']
                if blas_libraries[0].startswith('openblas'):
                    self.logger.debug('  BLAS library\t[OpenBLAS]\t\t({info})'.format(info=', '.join(blas_extra_info)))
                elif blas_libraries[0].startswith('blas'):
                    self.logger.debug('  BLAS library\t[BLAS/Atlas]\t\t({info})'.format(info=', '.join(blas_extra_info)))
                elif blas_libraries[0].startswith('mkl'):
                    self.logger.debug('  BLAS library\t[MKL]\t\t({info})'.format(info=', '.join(blas_extra_info)))

        # Select Keras backend
        os.environ["KERAS_BACKEND"] = self.learner_params.get_path('keras.backend', 'theano')


        if self.learner_params.get_path('keras.backend', 'theano') == 'theano':
            # Theano setup

            # Default flags
            flags = [
                #'ldflags=',
                'warn.round=False',
            ]

            # Set device
            if self.learner_params.get_path('keras.backend_parameters.device'):
                flags.append('device=' + self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))

                if self.show_extra_debug:
                    self.logger.debug('  Theano device \t[{device}]'.format(
                        device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))
                    )

            # Set floatX
            if self.learner_params.get_path('keras.backend_parameters.floatX'):
                flags.append('floatX=' + self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))

                if self.show_extra_debug:
                    self.logger.debug('  Theano floatX \t[{float}]'.format(
                        float=self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))
                    )

            # Set optimizer
            if self.learner_params.get_path('keras.backend_parameters.optimizer') is not None:
                if self.learner_params.get_path('keras.backend_parameters.optimizer') in ['fast_run', 'merge', 'fast_compile', 'None']:
                    flags.append('optimizer='+self.learner_params.get_path('keras.backend_parameters.optimizer'))
            else:
                flags.append('optimizer=None')
            if self.show_extra_debug:
                self.logger.debug('  Theano optimizer \t[{optimizer}]'.format(
                    optimizer=self.learner_params.get_path('keras.backend_parameters.optimizer', 'None'))
                )

            # Set fastmath for GPU mode only
            if self.learner_params.get_path('keras.backend_parameters.fastmath') is not None and self.learner_params.get_path('keras.backend_parameters.device', 'cpu') != 'cpu':
                if self.learner_params.get_path('keras.backend_parameters.fastmath', False):
                    flags.append('nvcc.fastmath=True')
                else:
                    flags.append('nvcc.fastmath=False')

                if self.show_extra_debug:
                    self.logger.debug('  NVCC fastmath \t[{flag}]'.format(
                        flag=str(self.learner_params.get_path('keras.backend_parameters.fastmath', False)))
                    )

            # Set OpenMP
            if self.learner_params.get_path('keras.backend_parameters.openmp') is not None:
                if self.learner_params.get_path('keras.backend_parameters.openmp', False):
                    flags.append('openmp=True')
                else:
                    flags.append('openmp=False')

                if self.show_extra_debug:
                    self.logger.debug('  OpenMP\t\t[{flag}]'.format(
                        flag=str(self.learner_params.get_path('keras.backend_parameters.openmp', False)))
                    )

            # Set environmental variable for Theano
            os.environ["THEANO_FLAGS"] = ','.join(flags)

        elif self.learner_params.get_path('keras.backend', 'tensorflow') == 'tensorflow':
            # Tensorflow setup

            # Set device
            if self.learner_params.get_path('keras.backend_parameters.device', 'cpu'):

                # In case of CPU disable visible GPU.
                if self.learner_params.get_path('keras.backend_parameters.device', 'cpu') == 'cpu':
                    os.environ["CUDA_VISIBLE_DEVICES"] = ''

                if self.show_extra_debug:
                    self.logger.debug('  Tensorflow device \t[{device}]'.format(
                        device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu')))

        else:
            message = '{name}: Keras backend not supported [backend].'.format(
                name=self.__class__.__name__,
                backend=self.learner_params.get_path('keras.backend')
            )
            self.logger.exception(message)
            raise AssertionError(message)


class SceneClassifier(LearnerContainer):
    """Scene classifier (Frame classifier / Multiclass - Singlelabel)"""

    def predict(self, feature_data, recognizer_params=None):
        """Predict scene label for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
        recognizer_params : DottedDict

        Returns
        -------
        str
            class label

        """

        if recognizer_params is None:
            recognizer_params = {}

        if not isinstance(recognizer_params, DottedDict):
            # Convert parameters to DottedDict
            recognizer_params = DottedDict(recognizer_params)

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame wise probabilities
        frame_probabilities = self._frame_probabilities(feature_data)

        # Accumulate probabilities
        if recognizer_params.get_path('frame_accumulation.enable', True):
            probabilities = self._accumulate_probabilities(probabilities=frame_probabilities,
                                                           accumulation_type=recognizer_params.get_path('frame_accumulation.type'))
        else:
            # Pass probabilities
            probabilities = frame_probabilities

        # Probability binarization
        if recognizer_params.get_path('frame_binarization.enable', True):
            if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                frame_decisions = numpy.argmax(
                    probabilities > recognizer_params.get_path('frame_binarization.threshold', 0.5),
                    axis=0
                )

            elif recognizer_params.get_path('frame_binarization.type') == 'frame_max':
                frame_decisions = numpy.argmax(probabilities, axis=0)

            else:
                message = '{name}: Unknown frame_binarization type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=recognizer_params.get_path('frame_binarization.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        # Decision making
        if recognizer_params.get_path('decision_making.enable', True):
            if recognizer_params.get_path('decision_making.type') == 'maximum':
                classification_result_id = numpy.argmax(probabilities)

            elif recognizer_params.get_path('decision_making.type') == 'majority_vote':
                counts = numpy.bincount(frame_decisions)
                classification_result_id = numpy.argmax(counts)

            else:
                message = '{name}: Unknown decision_making type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=recognizer_params.get_path('decision_making.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        return self.class_labels[classification_result_id]

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

    def _accumulate_probabilities(self, probabilities, accumulation_type='sum'):
        accumulated = numpy.ones(len(self.class_labels)) * -numpy.inf
        for row_id in range(0, probabilities.shape[0]):
            if accumulation_type == 'sum':
                accumulated[row_id] = numpy.sum(probabilities[row_id, :])
            elif accumulation_type == 'prod':
                accumulated[row_id] = numpy.prod(probabilities[row_id, :])
            elif accumulation_type == 'mean':
                accumulated[row_id] = numpy.mean(probabilities[row_id, :])
            else:
                message = '{name}: Unknown accumulation type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=accumulation_type
                )

                self.logger.exception(message)
                raise AssertionError(message)

        return accumulated

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            frame_count = data[audio_filename].feat[0].shape[0]
            pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
            roll = numpy.zeros((frame_count, len(self.class_labels)))
            roll[:, pos] = 1
            activity_matrix_dict[audio_filename] = roll
        return activity_matrix_dict

    def learn(self, data, annotations):
        message = '{name}: Implement learn function.'.format(
            name=self.__class__.__name__
        )

        self.logger.exception(message)
        raise AssertionError(message)


class SceneClassifierGMM(SceneClassifier):
    """Scene classifier with GMM"""
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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

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
    """Scene classifier with MLP"""
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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        training_files = sorted(list(annotations.keys()))  # Collect training files
        if self.learner_params.get_path('validation.enable', False):
            validation_files = self._generate_validation(
                annotations=annotations,
                validation_type=self.learner_params.get_path('validation.setup_source'),
                valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                seed=self.learner_params.get_path('validation.seed')
            )
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
        X_training = self.process_data(data=data, files=training_files)
        Y_training = self.process_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)

        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        # Process validation data
        if validation_files:
            X_validation = self.process_data(data=data, files=validation_files)
            Y_validation = self.process_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

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

        class FancyProgbarLogger(keras.callbacks.Callback):
            """Callback that prints metrics to stdout.
            """

            def __init__(self, callbacks=None, queue_length=10, metric=None, disable_progress_bar=False, log_progress=False):
                self.metric = metric
                self.disable_progress_bar = disable_progress_bar
                self.log_progress = log_progress
                self.timer = Timer()

            def on_train_begin(self, logs=None):
                self.logger = logging.getLogger(__name__)
                self.verbose = self.params['verbose']
                self.epochs = self.params['epochs']
                if self.log_progress:
                    self.logger.info('Starting training process')
                self.pbar = tqdm(total=self.epochs,
                                 file=sys.stdout,
                                 desc='           {0:>15s}'.format('Learn (epoch)'),
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar
                                 )

            def on_train_end(self, logs=None):
                self.pbar.close()

            def on_epoch_begin(self, epoch, logs=None):
                if self.log_progress:
                    self.logger.info('  Epoch %d/%d' % (epoch + 1, self.epochs))
                self.seen = 0
                self.timer.start()

            def on_batch_begin(self, batch, logs=None):
                if self.seen < self.params['samples']:
                    self.log_values = []

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                batch_size = logs.get('size', 0)
                self.seen += batch_size

                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                postfix = {
                    'train': None,
                    'validation': None,
                }
                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))
                        if self.metric and k.endswith(self.metric):
                            if k.startswith('val_'):
                                postfix['validation'] = '{:4.2f}'.format(logs[k] * 100.0)
                            else:
                                postfix['train'] = '{:4.2f}'.format(logs[k] * 100.0)
                self.timer.stop()
                if self.log_progress:
                    self.logger.info('                train={train}, validation={validation}, time={time}'.format(
                        train=postfix['train'],
                        validation=postfix['validation'],
                        time=self.timer.get_string())
                    )

                self.pbar.set_postfix(postfix)
                self.pbar.update(1)

        # Add model callbacks
        fancy_logger = FancyProgbarLogger(metric=self.learner_params.get_path('model.metrics')[0],
                                          disable_progress_bar=self.disable_progress_bar,
                                          log_progress=self.log_progress)

        # Callback list, always have FancyProgbarLogger
        callbacks = [fancy_logger]

        callback_params = self.learner_params.get_path('training.callbacks', [])
        if callback_params:
            for cp in callback_params:
                if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                    cp['parameters']['filepath'] = os.path.splitext(self.filename)[0] + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

                if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith('val_') and not self.learner_params.get_path('validation.enable', False):
                    message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] as there is no validation set.'.format(
                        name=self.__class__.__name__,
                        type=cp['type'],
                        monitor=cp.get('parameters').get('monitor')
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

                try:
                    # Get Callback class
                    CallbackClass = getattr(importlib.import_module("keras.callbacks"), cp['type'])

                    # Add callback to list
                    callbacks.append(CallbackClass(**cp.get('parameters', {})))

                except AttributeError:
                    message = '{name}: Invalid Keras callback type [{type}]'.format(
                        name=self.__class__.__name__,
                        type=cp['type']
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

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

        hist = self.model.fit(x=X_training,
                              y=Y_training,
                              batch_size=self.learner_params.get_path('training.batch_size', 1),
                              epochs=self.learner_params.get_path('training.epochs', 1),
                              validation_data=validation,
                              verbose=0,
                              shuffle=self.learner_params.get_path('training.shuffle', True),
                              callbacks=callbacks
                              )
        self['learning_history'] = hist.history

    def _frame_probabilities(self, feature_data):
        return self.model.predict(x=feature_data).T


class EventDetector(LearnerContainer):
    """Event detector (Frame classifier / Multiclass - Multilabel)"""
    @staticmethod
    def _contiguous_regions(activity_array):
        """Find contiguous regions from bool valued numpy.array.
        Transforms boolean values for each frame into pairs of onsets and offsets.
        Parameters
        ----------
        activity_array : numpy.array [shape=(t)]
            Event activity array, bool values
        Returns
        -------
        change_indices : numpy.ndarray [shape=(2, number of found changes)]
            Onset and offset indices pairs in matrix
        """

        # Find the changes in the activity_array
        change_indices = numpy.diff(activity_array).nonzero()[0]

        # Shift change_index with one, focus on frame after the change.
        change_indices += 1

        if activity_array[0]:
            # If the first element of activity_array is True add 0 at the beginning
            change_indices = numpy.r_[0, change_indices]

        if activity_array[-1]:
            # If the last element of activity_array is True, add the length of the array
            change_indices = numpy.r_[change_indices, activity_array.size]

        # Reshape the result into two columns
        return change_indices.reshape((-1, 2))

    def _slide_and_accumulate(self, input_probabilities, window_length, accumulation_type='sliding_sum'):
        # Lets keep the system causal and use look-back while smoothing (accumulating) likelihoods
        output_probabilities = copy.deepcopy(input_probabilities)
        for stop_id in range(0, input_probabilities.shape[0]):
            start_id = stop_id - window_length
            if start_id < 0:
                start_id = 0
            if start_id != stop_id:
                if accumulation_type == 'sliding_sum':
                    output_probabilities[start_id] = numpy.sum(input_probabilities[start_id:stop_id])
                elif accumulation_type == 'sliding_mean':
                    output_probabilities[start_id] = numpy.mean(input_probabilities[start_id:stop_id])
                elif accumulation_type == 'sliding_median':
                    output_probabilities[start_id] = numpy.median(input_probabilities[start_id:stop_id])
                else:
                    message = '{name}: Unknown slide and accumulate type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=accumulation_type
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                output_probabilities[start_id] = input_probabilities[start_id]

        return output_probabilities

    def _activity_processing(self, activity_vector, window_size, processing_type="median_filtering"):
        if processing_type == 'median_filtering':
            return scipy.signal.medfilt(volume=activity_vector, kernel_size=window_size)
        else:
            message = '{name}: Unknown activity processing type [{type}].'.format(
                name=self.__class__.__name__,
                type=processing_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

    def _get_target_matrix_dict(self, data, annotations):

        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            # Create event roll
            event_roll = EventRoll(metadata_container=annotations[audio_filename],
                                   label_list=self.class_labels,
                                   time_resolution=self.params.get_path('hop_length_seconds')
                                   )
            # Pad event roll to full length of the signal
            activity_matrix_dict[audio_filename] = event_roll.pad(length=data[audio_filename].feat[0].shape[0])

        return activity_matrix_dict

    def _generate_validation(self, annotations, validation_type='generated_scene_location_event_balanced',
                             valid_percentage=0.20, seed=None):

        self.set_seed(seed=seed)
        validation_files = []

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
                        self.logger.debug('  Valid sets found [{sets}]'.format(
                            sets=len(validation_set_MAE))
                        )

                        self.logger.debug('  Best fitting set ID={id}, Error={error:4.2}%'.format(
                            id=best_set_id,
                            error=validation_set_MAE[best_set_id]*100)
                        )

                        self.logger.debug('  Validation event counts in respect of all data:')
                        event_amount_percentages = validation_set_event_amounts[best_set_id] / (validation_set_event_amounts[best_set_id] + training_set_event_amounts[best_set_id])

                        self.logger.debug('  {event:<20s} | {amount:10s} '.format(
                            event='Event label',
                            amount='Validation amount (%)')
                        )

                        self.logger.debug('  {event:<20s} + {amount:10s} '.format(
                            event='-' * 20,
                            amount='-' * 20)
                        )

                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            self.logger.debug('  {event:<20s} | {amount:4.2f} '.format(
                                event=event_label,
                                amount=numpy.round(event_amount_percentages[event_label_id] * 100))
                            )
                        self.logger.debug('  ')
                else:
                    message = '{name}: Validation setup creation was not successful! Could not find a set with examples for each event class in both training and validation.'.format(
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
                self.logger.debug('  {event_label:<20s} | {amount:20s} '.format(
                    event_label='Event label',
                    amount='Validation amount, files (%)')
                )

                self.logger.debug('  {event_label:<20s} + {amount:20s} '.format(
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
                    self.logger.debug('  {event_label:<20s} | {amount:4.2f} '.format(
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

        return sorted(validation_files)


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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

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

        class_progress = tqdm(self.class_labels,
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


    def predict(self, feature_data, recognizer_params=None):
        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            # Evaluate positive and negative models
            if self['model'][event_label]['positive']:
                positive = self['model'][event_label]['positive'].score_samples(feature_data.feat[0])

                # Accumulate
                if recognizer_params.get_path('frame_accumulation.enable'):
                    positive = self._slide_and_accumulate(
                        input_probabilities=positive,
                        window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                        accumulation_type=recognizer_params.get_path('frame_accumulation.type')
                    )

            if self['model'][event_label]['negative']:
                negative = self['model'][event_label]['negative'].score_samples(feature_data.feat[0])

                # Accumulate
                if recognizer_params.get_path('frame_accumulation.enable'):
                    negative = self._slide_and_accumulate(
                        input_probabilities=negative,
                        window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                        accumulation_type=recognizer_params.get_path('frame_accumulation.type')
                    )
            if self['model'][event_label]['positive'] and self['model'][event_label]['negative']:
                # Likelihood ratio
                frame_probabilities = positive - negative
            elif self['model'][event_label]['positive'] is None and self['model'][event_label]['negative'] is not None:
                # Likelihood ratio
                frame_probabilities = -negative

            elif self['model'][event_label]['positive'] is not None and self['model'][event_label]['negative'] is None:
                # Likelihood ratio
                frame_probabilities = positive

            # Binarization
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities > recognizer_params.get_path('frame_binarization.threshold', 0.0)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise AssertionError(message)

            # Get events
            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')

            # Add events
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0],
                                             'event_offset': event[1],
                                             'event_label': event_label}))

        return MetaDataContainer(results)


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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

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

        class_progress = tqdm(self.class_labels,
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

    def _frame_probabilities(self, feature_data, accumulation_window_length_frames=None):
        probabilities = numpy.ones((len(self.class_labels), feature_data.shape[0])) * -numpy.inf
        for event_id, event_label in enumerate(self.class_labels):
            positive = self['model'][event_label]['positive'].score_samples(feature_data.feat[0])[0]
            negative = self['model'][event_label]['negative'].score_samples(feature_data.feat[0])[0]
            if accumulation_window_length_frames:
                positive = self._slide_and_accumulate(input_probabilities=positive, window_length=accumulation_window_length_frames)
                negative = self._slide_and_accumulate(input_probabilities=negative, window_length=accumulation_window_length_frames)
            probabilities[event_id, :] = positive - negative

        return probabilities

    def predict(self, feature_data, recognizer_params=None):
        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)

        frame_probabilities = self._frame_probabilities(
            feature_data=feature_data,
            accumulation_window_length_frames=recognizer_params.get_path('frame_accumulation.window_length_frames')
        )

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities[event_id, :] > recognizer_params.get_path('frame_binarization.threshold', 0.0)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )
                    self.logger.exception(message)
                    raise AssertionError(message)
            else:
                message = '{name}: No frame_binarization enabled.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise AssertionError(message)

            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0], 'event_offset': event[1], 'event_label': event_label}))

        return MetaDataContainer(results)


class EventDetectorMLP(EventDetector, KerasMixin):
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

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(annotations=annotations,
                                                             validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                                                             valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                                                             seed=self.learner_params.get_path('validation.seed'),
                                                             )

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
        X_training = self.process_data(data=data, files=training_files)
        Y_training = self.process_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)

        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        # Process validation data
        if validation_files:
            X_validation = self.process_data(data=data, files=validation_files)
            Y_validation = self.process_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

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

                self.logger.debug('  Positives items \t[{positives:d}]\t({perc:.2f} %)'.format(
                    positives=len(positive_examples_id),
                    perc=len(positive_examples_id)/float(len(positive_examples_id)+len(negative_examples_id))*100
                ))
                self.logger.debug('  Negatives items \t[{negatives:d}]\t({perc:.2f} %)'.format(
                    negatives=len(negative_examples_id),
                    perc=len(negative_examples_id) / float(len(positive_examples_id) + len(negative_examples_id)) * 100
                ))

                self.logger.debug('  Class weights \t[{weights}]\t'.format(weights=class_weight))

        class FancyProgbarLogger(keras.callbacks.Callback):
            """Callback that prints metrics to stdout.
            """

            def __init__(self, callbacks=None, queue_length=10, metric=None, disable_progress_bar=False, log_progress=False):
                if isinstance(metric, str):
                    self.metric = metric
                elif callable(metric):
                    self.metric = metric.__name__
                self.disable_progress_bar = disable_progress_bar
                self.log_progress = log_progress
                self.timer = Timer()

            def on_train_begin(self, logs=None):
                self.logger = logging.getLogger(__name__)
                self.verbose = self.params['verbose']
                self.epochs = self.params['epochs']
                if self.log_progress:
                    self.logger.info('Starting training process')
                self.pbar = tqdm(total=self.epochs,
                                 file=sys.stdout,
                                 desc='           {0:>15s}'.format('Learn (epoch)'),
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar
                                 )

            def on_train_end(self, logs=None):
                self.pbar.close()

            def on_epoch_begin(self, epoch, logs=None):
                if self.log_progress:
                    self.logger.info('  Epoch %d/%d' % (epoch + 1, self.epochs))
                self.seen = 0
                self.timer.start()

            def on_batch_begin(self, batch, logs=None):
                if self.seen < self.params['samples']:
                    self.log_values = []

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                batch_size = logs.get('size', 0)
                self.seen += batch_size

                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                postfix = {
                    'train': None,
                    'validation': None,
                }
                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))
                        if self.metric and k.endswith(self.metric):
                            if k.startswith('val_'):
                                postfix['validation'] = '{:4.4f}'.format(logs[k])
                            else:
                                postfix['train'] = '{:4.4f}'.format(logs[k])
                self.timer.stop()
                if self.log_progress:
                    self.logger.info('                train={train}, validation={validation}, time={time}'.format(
                        train=postfix['train'],
                        validation=postfix['validation'],
                        time=self.timer.get_string())
                    )

                self.pbar.set_postfix(postfix)
                self.pbar.update(1)

        # Add model callbacks
        fancy_logger = FancyProgbarLogger(
            metric=self.learner_params.get_path('model.metrics')[0],
            disable_progress_bar=self.disable_progress_bar,
            log_progress=self.log_progress
        )

        callbacks = [fancy_logger]
        callback_params = self.learner_params.get_path('training,callbacks', [])
        if callback_params:
            for cp in callback_params:
                if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                    cp['parameters']['filepath'] = os.path.splitext(self.filename)[0] + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

                if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith('val_') and not self.learner_params.get_path('validation.enable', False):
                    message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] as there is no validation set.'.format(
                        name=self.__class__.__name__,
                        type=cp['type'],
                        monitor=cp.get('parameters').get('monitor')
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

                try:
                    CallbackClass = getattr(importlib.import_module("keras.callbacks"), cp['type'])
                    callbacks.append(CallbackClass(**cp.get('parameters', {})))

                except AttributeError:
                    message = '{name}: Invalid Keras callback type [{type}]'.format(
                        name=self.__class__.__name__,
                        type=cp['type']
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

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
            callbacks=callbacks,
            class_weight=class_weight
        )

        self['learning_history'] = hist.history

    def predict(self, feature_data, recognizer_params=None):

        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]
        frame_probabilities = self.model.predict(x=feature_data).T

        if recognizer_params.get_path('frame_accumulation.enable'):
            for event_id, event_label in enumerate(self.class_labels):
                frame_probabilities[event_id, :] = self._slide_and_accumulate(
                    input_probabilities=frame_probabilities[event_id, :],
                    window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                    accumulation_type=recognizer_params.get_path('frame_accumulation.type'),
                )

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities[event_id, :] > recognizer_params.get_path('frame_binarization.threshold', 0.5)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise AssertionError(message)

            if recognizer_params.get_path('event_activity_processing.enable'):
                event_activity = self._activity_processing(activity_vector=event_activity,
                                                           window_size=recognizer_params.get_path('event_activity_processing.window_length_frames'))

            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0], 'event_offset': event[1], 'event_label': event_label}))

        return MetaDataContainer(results)

