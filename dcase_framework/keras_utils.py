#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keras utils
===========

Utility classes related to Keras.

KerasMixin
^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    KerasMixin.create_model
    KerasMixin.create_callback_list
    KerasMixin.create_external_metric_evaluators
    KerasMixin.process_data
    KerasMixin.process_activity
    KerasMixin.keras_model_exists
    KerasMixin.log_model_summary
    KerasMixin.plot_model
    KerasMixin.get_processing_interval

BaseCallback
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    BaseCallback

ProgressLoggerCallback
^^^^^^^^^^^^^^^^^^^^^^

Keras callback to store metrics with tqdm progress bar or logging interface. Implements Keras Callback API.

This callback is very similar to standard ``ProgbarLogger`` Keras callback, however it adds support for
logging interface and tqdm based progress bars, and external metrics
(metrics calculated outside Keras training process).

.. autosummary::
    :toctree: generated/

    ProgressLoggerCallback

ProgressPlotterCallback
^^^^^^^^^^^^^^^^^^^^^^^

Keras callback to plot progress during the training process and save final progress into figure.
Implements Keras Callback API.

.. autosummary::
    :toctree: generated/

    ProgressPlotterCallback

StopperCallback
^^^^^^^^^^^^^^^

Keras callback to stop training when improvement has not seen in specified amount of epochs.
Implements Keras Callback API.

This Callback is very similar to standard ``EarlyStopping`` Keras callback, however it adds support for
external metrics (metrics calculated outside Keras training process).

.. autosummary::
    :toctree: generated/

    StopperCallback

StasherCallback
^^^^^^^^^^^^^^^

Keras callback to monitor training process and store best model. Implements Keras Callback API.

This callback is very similar to standard ``ModelCheckpoint`` Keras callback, however it adds support for
external metrics (metrics calculated outside Keras training process).

.. autosummary::
    :toctree: generated/

    StasherCallback

BaseDataGenerator
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    BaseDataGenerator
    BaseDataGenerator.input_size
    BaseDataGenerator.data_size
    BaseDataGenerator.steps_count
    BaseDataGenerator.info

FeatureGenerator
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    FeatureGenerator
    FeatureGenerator.generator

"""

import os
import sys
import logging
import numpy
import copy
import importlib
import collections
from tqdm import tqdm
from six import iteritems

from .containers import DottedDict
from .utils import SuppressStdoutAndStderr, Timer, SimpleMathStringEvaluator, get_parameter_hash
from .features import FeatureContainer
from .metadata import EventRoll
from .data import DataBuffer


class KerasMixin(object):
    """Class Mixin for Keras based learner containers.

    """

    def __getstate__(self):
        data = {}
        excluded_fields = ['model']

        for item in self:
            if item not in excluded_fields and self.get(item):
                data[item] = copy.deepcopy(self.get(item))
        data['model'] = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return data

    def keras_model_exists(self):
        """Check that keras model exists on disk

        Returns
        -------
        bool

        """

        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return os.path.isfile(self.filename) and os.path.isfile(keras_model_filename)

    def log_model_summary(self):
        """Prints model summary to the logging interface.

        Similar to Keras model summary
        """

        layer_name_map = {
            'BatchNormalization': 'BatchNorm',
        }
        import keras.backend as keras_backend

        self.logger.debug('  Model summary')
        self.logger.debug(
            '    {type:<15s} | {out:20s} | {param:6s}  | {name:21s}  | {conn:27s} | {act:7s} | {init:7s}'.format(
                type='Layer type',
                out='Output',
                param='Param',
                name='Name',
                conn='Connected to',
                act='Activ.',
                init='Init')
        )

        self.logger.debug(
            '    {type:<15s} + {out:20s} + {param:6s}  + {name:21s}  + {conn:27s} + {act:7s} + {init:6s}'.format(
                type='-' * 15,
                out='-' * 20,
                param='-' * 6,
                name='-' * 21,
                conn='-' * 27,
                act='-' * 7,
                init='-' * 6)
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

            config = DottedDict(layer.get_config())
            layer_name = layer.__class__.__name__
            if layer_name in layer_name_map:
                layer_name = layer_name_map[layer_name]

            if config.get_path('kernel_initializer.class_name') == 'VarianceScaling':
                init = str(config.get_path('kernel_initializer.config.distribution', '---'))
            elif config.get_path('kernel_initializer.class_name') == 'RandomUniform':
                init = 'uniform'
            else:
                init = '---'

            self.logger.debug(
                '    {type:<15s} | {shape:20s} | {params:6s}  | {name:21s}  | {connected:27s} | {activation:7s} | {init:7s}'.format(
                    type=layer_name,
                    shape=str(layer.output_shape),
                    params=str(layer.count_params()),
                    name=str(layer.name),
                    connected=str(connections[0]) if len(connections) > 0 else '---',
                    activation=str(config.get('activation', '---')),
                    init=init,

                )
            )

        trainable_count = int(
            numpy.sum([keras_backend.count_params(p) for p in set(self.model.trainable_weights)])
        )

        non_trainable_count = int(
            numpy.sum([keras_backend.count_params(p) for p in set(self.model.non_trainable_weights)])
        )

        self.logger.debug('  ')
        self.logger.debug('  Parameters')
        self.logger.debug('    Trainable\t[{param_count:,}]'.format(param_count=int(trainable_count)))
        self.logger.debug('    Non-Trainable\t[{param_count:,}]'.format(param_count=int(non_trainable_count)))
        self.logger.debug(
            '    Total\t\t[{param_count:,}]'.format(param_count=int(trainable_count + non_trainable_count)))
        self.logger.debug('  ')

    def plot_model(self, filename='model.png', show_shapes=True, show_layer_names=True):
        """Plots model topology
        """

        from keras.utils.visualize_util import plot
        plot(self.model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)

    def prepare_data(self, data, files, processor='default'):
        """Concatenate feature data into one feature matrix

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        files : list of str
            List of filenames
        processor : str ('default', 'training')
            Data processor selector
            Default value 'default'

        Returns
        -------
        numpy.ndarray
            Features concatenated

        """

        if self.learner_params.get_path('input_sequencer.enable'):
            processed_data = []
            for item in files:
                if processor == 'training':
                    processed_data.append(
                        self.data_processor_training.process_data(
                            data=data[item].feat[0]
                        )
                    )

                else:
                    processed_data.append(
                        self.data_processor.process_data(
                            data=data[item].feat[0]
                        )
                    )

            return numpy.concatenate(processed_data)

        else:
            return numpy.vstack([data[x].feat[0] for x in files])

    def prepare_activity(self, activity_matrix_dict, files, processor='default'):
        """Concatenate activity matrices into one activity matrix

        Parameters
        ----------
        activity_matrix_dict : dict of binary matrices
            Meta data
        files : list of str
            List of filenames
        processor : str ('default', 'training')
            Data processor selector
            Default value 'default'
        Returns
        -------
        numpy.ndarray
            Activity matrix
        """

        if self.learner_params.get_path('input_sequencer.enable'):
            processed_activity = []
            for item in files:
                if processor == 'training':
                    processed_activity.append(
                        self.data_processor_training.process_activity_data(
                            activity_data=activity_matrix_dict[item]
                        )
                    )

                else:
                    processed_activity.append(
                        self.data_processor.process_activity_data(
                            activity_data=activity_matrix_dict[item]
                        )
                    )

            return numpy.concatenate(processed_activity)

        else:
            return numpy.vstack([activity_matrix_dict[x] for x in files])

    def create_model(self, input_shape):
        """Create sequential Keras model
        """

        from keras.models import Sequential
        self.model = Sequential()

        tuple_fields = [
            'input_shape',
            'kernel_size',
            'pool_size',
            'dims',
            'target_shape'
        ]

        # Get model config parameters
        model_params = copy.deepcopy(self.learner_params.get_path('model.config'))

        # Get constants for model
        constants = copy.deepcopy(self.learner_params.get_path('model.constants', {}))
        constants['CLASS_COUNT'] = len(self.class_labels)
        constants['FEATURE_VECTOR_LENGTH'] = input_shape
        if self.learner_params.get_path('input_sequencer.frames'):
            constants['INPUT_SEQUENCE_LENGTH'] = self.learner_params.get_path('input_sequencer.frames')

        def process_field(value, constants_dict):
            math_eval = SimpleMathStringEvaluator()

            if isinstance(value, str):
                # String field
                if value in constants_dict:
                    return constants_dict[value]
                elif len(value.split()) > 1:
                    sub_fields = value.split()
                    for subfield_id, subfield in enumerate(sub_fields):
                        if subfield in constants_dict:
                            sub_fields[subfield_id] = str(constants_dict[subfield])
                    return math_eval.eval(''.join(sub_fields))
                else:
                    return value

            elif isinstance(value, list):
                processed_value_list = []
                for item_id, item in enumerate(value):
                    processed_value_list.append(process_field(value=item, constants_dict=constants_dict))
                return processed_value_list
            else:
                return value

        # Inject constant into constants with equations
        for field in list(constants.keys()):
            constants[field] = process_field(value=constants[field], constants_dict=constants)

        # Setup layers
        for layer_id, layer_setup in enumerate(model_params):
            # Get layer parameters
            layer_setup = DottedDict(layer_setup)
            if 'config' not in layer_setup:
                layer_setup['config'] = {}

            # Get layer class
            try:
                layer_class = getattr(
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

            # Inject constants
            for config_field in list(layer_setup['config'].keys()):
                layer_setup['config'][config_field] = process_field(
                    value=layer_setup['config'][config_field],
                    constants_dict=constants
                )

            # Convert lists into tuples
            for field in tuple_fields:
                if field in layer_setup['config']:
                    layer_setup['config'][field] = tuple(layer_setup['config'][field])

            # Inject input shape for Input layer if not given
            if layer_id == 0 and layer_setup.get_path('config.input_shape') is None:
                # Set input layer dimension for the first layer if not set
                layer_setup['config']['input_shape'] = (input_shape,)

            if 'wrapper' in layer_setup:
                # Get layer wrapper class
                try:
                    wrapper_class = getattr(
                        importlib.import_module("keras.layers"),
                        layer_setup['wrapper']
                    )

                except AttributeError:
                    message = '{name}: Invalid Keras layer wrapper type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=layer_setup['wrapper']
                    )
                    self.logger.exception(message)
                    raise AttributeError(message)

                wrapper_parameters = layer_setup.get('config_wrapper', {})

                if layer_setup.get('config'):
                    self.model.add(
                        wrapper_class(layer_class(**dict(layer_setup.get('config'))), **dict(wrapper_parameters)))
                else:
                    self.model.add(wrapper_class(layer_class(), **dict(wrapper_parameters)))

            else:
                if layer_setup.get('config'):
                    self.model.add(layer_class(**dict(layer_setup.get('config'))))
                else:
                    self.model.add(layer_class())

        # Get Optimizer class
        try:
            optimizer_class = getattr(
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
            optimizer=optimizer_class(**dict(self.learner_params.get_path('model.optimizer.parameters', {}))),
            metrics=self.learner_params.get_path('model.metrics')
        )

    def create_callback_list(self):
        """Create list of Keras callbacks
        """

        callbacks = []

        # Fetch processing interval
        processing_interval = self.get_processing_interval()

        # Collect all external metrics
        external_metrics = collections.OrderedDict()
        if self.learner_params.get_path('training.epoch_processing.enable'):
            if self.learner_params.get_path('validation.enable') and self.learner_params.get_path(
                    'training.epoch_processing.external_metrics.enable'):
                for metric in self.learner_params.get_path('training.epoch_processing.external_metrics.metrics'):
                    current_metric_name = metric.get('name')
                    current_metric_label = metric.get('label', current_metric_name.split('.')[-1])

                    external_metrics[current_metric_label] = current_metric_name

        # ProgressLoggerCallback
        from dcase_framework.keras_utils import ProgressLoggerCallback
        callbacks.append(
            ProgressLoggerCallback(
                metric=self.learner_params.get_path('model.metrics')[0],
                loss=self.learner_params.get_path('model.loss'),
                disable_progress_bar=self.disable_progress_bar,
                log_progress=self.log_progress,
                epochs=self.learner_params.get_path('training.epochs'),
                close_progress_bar=not self.learner_params.get_path('training.epoch_processing.enable'),
                manual_update=self.learner_params.get_path('training.epoch_processing.enable'),
                manual_update_interval=processing_interval,
                external_metric_labels=external_metrics
            )
        )

        # Add model callbacks
        for cp in self.learner_params.get_path('training.callbacks', []):
            cp_params = DottedDict(cp.get('parameters', {}))
            if cp['type'] == 'Plotter':
                from dcase_framework.keras_utils import ProgressPlotterCallback
                callbacks.append(
                    ProgressPlotterCallback(
                        filename=os.path.splitext(self.filename)[0] + '.' + cp_params.get('output_format', 'pdf'),
                        metric=self.learner_params.get_path('model.metrics')[0],
                        loss=self.learner_params.get_path('model.loss'),
                        disable_progress_bar=self.disable_progress_bar,
                        log_progress=self.log_progress,
                        epochs=self.learner_params.get_path('training.epochs'),
                        close_progress_bar=not self.learner_params.get_path('training.epoch_processing.enable'),
                        manual_update=self.learner_params.get_path('training.epoch_processing.enable'),
                        interactive=cp_params.get('interactive', True),
                        save=cp_params.get('save', True),
                        focus_span=cp_params.get('focus_span'),
                        plotting_rate=cp_params.get('plotting_rate'),
                        external_metric_labels=external_metrics
                    )
                )

            elif cp['type'] == 'Stopper':
                from dcase_framework.keras_utils import StopperCallback
                callbacks.append(
                    StopperCallback(
                        epochs=self.learner_params.get_path('training.epochs'),
                        manual_update=self.learner_params.get_path('training.epoch_processing.enable'),
                        external_metric_labels=external_metrics,
                        **cp_params
                    )
                )

            elif cp['type'] == 'Stasher':
                from dcase_framework.keras_utils import StasherCallback
                callbacks.append(
                    StasherCallback(
                        epochs=self.learner_params.get_path('training.epochs'),
                        manual_update=self.learner_params.get_path('training.epoch_processing.enable'),
                        external_metric_labels=external_metrics,
                        **cp_params
                    )
                )

            else:
                # Keras standard callbacks
                if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                    cp['parameters']['filepath'] = os.path.splitext(self.filename)[0] + \
                                                   '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

                if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith('val_') \
                        and not self.learner_params.get_path('validation.enable', False):

                    message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] ' \
                              'as there is no validation set.'.format(name=self.__class__.__name__,
                                                                      type=cp['type'],
                                                                      monitor=cp.get('parameters').get('monitor')
                                                                      )

                    self.logger.exception(message)
                    raise AttributeError(message)

                try:
                    callback_class = getattr(importlib.import_module("keras.callbacks"), cp['type'])
                    callbacks.append(callback_class(**cp_params))

                except AttributeError:
                    message = '{name}: Invalid Keras callback type [{type}]'.format(
                        name=self.__class__.__name__,
                        type=cp['type']
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)
        return callbacks

    def create_external_metric_evaluators(self):
        """Create external metric evaluators
        """

        # Initialize external metrics
        external_metric_evaluators = collections.OrderedDict()
        if self.learner_params.get_path('training.epoch_processing.enable'):
            if self.learner_params.get_path('validation.enable') and self.learner_params.get_path(
                    'training.epoch_processing.external_metrics.enable'):
                import sed_eval

                for metric in self.learner_params.get_path('training.epoch_processing.external_metrics.metrics'):
                    # Current metric info
                    current_metric_evaluator = metric.get('evaluator')
                    current_metric_name = metric.get('name')
                    current_metric_params = metric.get('parameters', {})
                    current_metric_label = metric.get('label', current_metric_name.split('.')[-1])

                    # Initialize sed_eval evaluators
                    if current_metric_evaluator == 'sed_eval.scene':
                        evaluator = sed_eval.scene.SceneClassificationMetrics(
                            scene_labels=self.class_labels,
                            **current_metric_params
                        )

                    elif (current_metric_evaluator == 'sed_eval.segment_based' or
                          current_metric_evaluator == 'sed_eval.sound_event.segment_based'):
                        evaluator = sed_eval.sound_event.SegmentBasedMetrics(
                            event_label_list=self.class_labels,
                            **current_metric_params
                        )

                    elif (current_metric_evaluator == 'sed_eval.event_based' or
                          current_metric_evaluator == 'sed_eval.sound_event.event_based'):
                        evaluator = sed_eval.sound_event.EventBasedMetrics(
                            event_label_list=self.class_labels,
                            **current_metric_params
                        )

                    else:
                        message = '{name}: Unknown target metric [{metric}].'.format(
                            name=self.__class__.__name__,
                            metric=current_metric_name
                        )
                        self.logger.exception(message)
                        raise AssertionError(message)

                    # Check evaluator API
                    if (not hasattr(evaluator, 'reset') or
                       not hasattr(evaluator, 'evaluate') or
                       not hasattr(evaluator, 'results')):
                        if current_metric_evaluator.startswith('sed_eval'):
                            message = '{name}: wrong version of sed_eval for [{current_metric_evaluator}::{current_metric_name}], update sed_eval to latest version'.format(
                                name=self.__class__.__name__,
                                current_metric_evaluator=current_metric_evaluator,
                                current_metric_name=current_metric_name
                            )

                            self.logger.exception(message)
                            raise ValueError(message)

                        else:
                            message = '{name}: Evaluator has invalid API [{current_metric_evaluator}::{current_metric_name}]'.format(
                                name=self.__class__.__name__,
                                current_metric_evaluator=current_metric_evaluator,
                                current_metric_name=current_metric_name
                            )

                            self.logger.exception(message)
                            raise ValueError(message)

                    # Form unique name for metric, to allow multiple similar metrics with different parameters
                    metric_id = get_parameter_hash(metric)

                    # Metric data container
                    metric_data = {
                        'evaluator_name': current_metric_evaluator,
                        'name': current_metric_name,
                        'params': current_metric_params,
                        'label': current_metric_label,
                        'path': current_metric_name,
                        'evaluator': evaluator,
                    }
                    external_metric_evaluators[metric_id] = metric_data

        return external_metric_evaluators

    def get_processing_interval(self):
        """Processing interval
        """

        processing_interval = 1
        if self.learner_params.get_path('training.epoch_processing.enable'):
            if self.learner_params.get_path('training.epoch_processing.external_metrics.enable'):
                processing_interval = self.learner_params.get_path(
                    'training.epoch_processing.external_metrics.evaluation_interval', 1)

        return processing_interval

    def _after_load(self, to_return=None):
        with SuppressStdoutAndStderr():
            # Setup Keras if not yet set up. This is needed as keras has tensorflow as default backend, and this will
            # give error if it is not installed and theano is not set up as backend.
            self._setup_keras()

            from keras.models import load_model

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
        """Setup keras backend and parameters
        """

        if not hasattr(self, 'keras_setup_done') or not self.keras_setup_done:
            # Get BLAS library associated to numpy
            if numpy.__config__.blas_opt_info and 'libraries' in numpy.__config__.blas_opt_info:
                blas_libraries = numpy.__config__.blas_opt_info['libraries']
            else:
                blas_libraries = ['']

            blas_extra_info = []
            # Set backend and parameters before importing keras
            if self.show_extra_debug:
                self.logger.debug('  Keras')
                self.logger.debug('    Backend \t[{backend}]'.format(
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
                if numpy.__config__.blas_opt_info and 'libraries' in numpy.__config__.blas_opt_info:
                    blas_libraries = numpy.__config__.blas_opt_info['libraries']
                    if blas_libraries[0].startswith('openblas'):
                        self.logger.debug('    BLAS library\t[OpenBLAS]\t\t({info})'.format(
                            info=', '.join(blas_extra_info))
                        )

                    elif blas_libraries[0].startswith('blas'):
                        self.logger.debug(
                            '  BLAS library\t[BLAS/Atlas]\t\t({info})'.format(
                                info=', '.join(blas_extra_info))
                        )

                    elif blas_libraries[0].startswith('mkl'):
                        self.logger.debug('    BLAS library\t[MKL]\t\t({info})'.format(
                            info=', '.join(blas_extra_info))
                        )

            # Select Keras backend
            os.environ["KERAS_BACKEND"] = self.learner_params.get_path('keras.backend', 'theano')

            if self.learner_params.get_path('keras.backend', 'theano') == 'theano':
                # Theano setup
                if self.show_extra_debug:
                    self.logger.debug('  Theano')
                # Default flags
                flags = [
                    # 'ldflags=',
                    'warn.round=False',
                ]

                # Set device
                if self.learner_params.get_path('keras.backend_parameters.device'):
                    flags.append('device=' + self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))

                    if self.show_extra_debug:
                        self.logger.debug('    Device \t\t[{device}]'.format(
                            device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))
                        )

                # Set floatX
                if self.learner_params.get_path('keras.backend_parameters.floatX'):
                    flags.append('floatX=' + self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))

                    if self.show_extra_debug:
                        self.logger.debug('    floatX \t\t[{float}]'.format(
                            float=self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))
                        )

                # Set optimizer
                if self.learner_params.get_path('keras.backend_parameters.optimizer') is not None:
                    if self.learner_params.get_path('keras.backend_parameters.optimizer') in ['fast_run', 'merge',
                                                                                              'fast_compile', 'None']:
                        flags.append('optimizer=' + self.learner_params.get_path('keras.backend_parameters.optimizer'))

                if self.show_extra_debug:
                    self.logger.debug('    Optimizer \t[{optimizer}]'.format(
                        optimizer=self.learner_params.get_path('keras.backend_parameters.optimizer', 'None'))
                    )

                # Set fastmath for GPU mode only
                if self.learner_params.get_path('keras.backend_parameters.fastmath') and self.learner_params.get_path(
                        'keras.backend_parameters.device', 'cpu') != 'cpu':
                    if self.learner_params.get_path('keras.backend_parameters.fastmath', False):
                        flags.append('nvcc.fastmath=True')
                    else:
                        flags.append('nvcc.fastmath=False')

                    if self.show_extra_debug:
                        self.logger.debug('    NVCC fastmath \t[{flag}]'.format(
                            flag=str(self.learner_params.get_path('keras.backend_parameters.fastmath', False)))
                        )

                # Set OpenMP
                if self.learner_params.get_path('keras.backend_parameters.openmp') is not None:
                    if self.learner_params.get_path('keras.backend_parameters.openmp', False):
                        flags.append('openmp=True')
                    else:
                        flags.append('openmp=False')

                    if self.show_extra_debug:
                        self.logger.debug('    OpenMP\t\t[{flag}]'.format(
                            flag=str(self.learner_params.get_path('keras.backend_parameters.openmp', False)))
                        )

                # Set environmental variable for Theano
                os.environ["THEANO_FLAGS"] = ','.join(flags)

            elif self.learner_params.get_path('keras.backend', 'tensorflow') == 'tensorflow':
                # Tensorflow setup
                if self.show_extra_debug:
                    self.logger.debug('  Tensorflow')
                # Set device
                if self.learner_params.get_path('keras.backend_parameters.device', 'cpu'):

                    # In case of CPU disable visible GPU.
                    if self.learner_params.get_path('keras.backend_parameters.device', 'cpu') == 'cpu':
                        os.environ["CUDA_VISIBLE_DEVICES"] = ''

                    if self.show_extra_debug:
                        self.logger.debug('    Device \t\t[{device}]'.format(
                            device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu')))

            else:
                message = '{name}: Keras backend not supported [backend].'.format(
                    name=self.__class__.__name__,
                    backend=self.learner_params.get_path('keras.backend')
                )
                self.logger.exception(message)
                raise AssertionError(message)
            if self.show_extra_debug:
                self.logger.debug('  ')

        self.keras_setup_done = True


class BaseCallback(object):
    """Base class for Callbacks
    """
    def __init__(self, *args, **kwargs):
        self.params = None
        self.model = None

        self.verbose = kwargs.get('verbose', True)
        self.manual_update = kwargs.get('manual_update', False)

        self.epochs = kwargs.get('epochs')
        self.epoch = 0

        self.external_metric_labels = kwargs.get('external_metric_labels', collections.OrderedDict())
        self.external_metric = collections.OrderedDict()

        self.keras_metrics = [
            'binary_accuracy',
            'categorical_accuracy',
            'sparse_categorical_accuracy',
            'top_k_categorical_accuracy'
        ]

        self.logger = logging.getLogger(__name__)

    def set_model(self, model):
        self.model = model

    def set_params(self, params):
        self.params = params

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def update(self):
        pass

    def add_external_metric(self, metric_label):
        pass

    def set_external_metric_value(self, metric_label, metric_value):
        pass

    def get_operator(self, metric):
        metric = metric.lower()
        if metric.endswith('error_rate') or metric.endswith('er'):
            return numpy.less

        elif (metric.endswith('f_measure') or
             metric.endswith('fmeasure') or
             metric.endswith('fscore') or
             metric.endswith('f-score')):

            return numpy.greater

        elif metric.endswith('accuracy') or metric.endswith('acc'):
            return numpy.greater

        else:
            return numpy.less


class ProgressLoggerCallback(BaseCallback):
    """Keras callback to store metrics with tqdm progress bar or logging interface. Implements Keras Callback API.

    This callback is very similar to standard ``ProgbarLogger`` Keras callback, however it adds support for logging
    interface and tqdm based progress bars, and external metrics (metrics calculated outside Keras training process).

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        epochs : int
            Total amount of epochs

        metric : str
            Metric name

        manual_update : bool
            Manually update callback, use this to when injecting external metrics
            Default value True

        manual_update_interval : int
            Epoch interval for manual update, used anticipate updates
            Default value 1

        disable_progress_bar : bool
            Disable tqdm based progress bar
            Default value False

        close_progress_bar : bool
            Close tqdm progress bar on training end
            Default value True

        log_progress : bool
            Print progress into logging interface
            Default value False

        external_metric_labels : dict or OrderedDict
            Dictionary with

        """

        super(ProgressLoggerCallback, self).__init__(*args, **kwargs)
        if isinstance(kwargs.get('metric'), str):
            self.metric = kwargs.get('metric')
        elif callable(kwargs.get('metric')):
            self.metric = kwargs.get('metric').__name__

        self.loss = kwargs.get('loss')
        self.disable_progress_bar = kwargs.get('disable_progress_bar', False)
        self.close_progress_bar = kwargs.get('close_progress_bar', True)
        self.manual_update_interval = kwargs.get('manual_update_interval', 1)

        self.log_progress = kwargs.get('log_progress', False)
        self.timer = Timer()

        self.progress_bar = None
        self.validation_data = None
        self.seen = 0
        self.log_values = []
        self.logger = logging.getLogger(__name__)

        self.postfix = collections.OrderedDict()
        self.postfix['l_tra'] = None
        self.postfix['l_val'] = None
        self.postfix['m_tra'] = None
        self.postfix['m_val'] = None

        self.data = {
            'l_tra': numpy.empty((self.epochs,)),
            'l_val': numpy.empty((self.epochs,)),
            'm_tra': numpy.empty((self.epochs,)),
            'm_val': numpy.empty((self.epochs,)),
        }
        self.data['l_tra'][:] = numpy.nan
        self.data['l_val'][:] = numpy.nan
        self.data['m_tra'][:] = numpy.nan
        self.data['m_val'][:] = numpy.nan

        for metric_label in self.external_metric_labels:
            self.data[metric_label] = numpy.empty((self.epochs,))
            self.data[metric_label][:] = numpy.nan

        self.header_show = False
        self.last_update_epoch = 0

    def on_train_begin(self, logs=None):
        if self.epochs is None:
            self.epochs = self.params['epochs']

        if self.log_progress and not self.header_show:
            # Show header only once
            self.header_show = True

            self.logger.info('  Training')
            header_extra1 = '    {epoch:<5s} | {loss:<19s} | {metric:<19s} | '.format(
                epoch=' '*5,
                loss='Loss',
                metric='Metric',
                validation=' '*8,
            )
            if self.external_metric_labels:
                line = '{external_value:<'+str(12*len(self.external_metric_labels))+'s} '
                header_extra1 += line.format(
                    external_value='External metric',
                )
            header_extra1 += '{time:<15s}'.format(
                time=' '*15,
            )
            loss_label = self.loss
            if len(loss_label) > 19:
                loss_label = loss_label[0:17]+'..'

            metric_label = self.metric
            if len(metric_label) > 19:
                metric_label = metric_label[0:17]+'..'

            header_extra2 = '    {epoch:<5s} | {loss:<19s} | {metric:<19s} | '.format(
                epoch=' '*5,
                loss=loss_label,
                metric=metric_label,
                validation=' '*8,
            )
            if self.external_metric_labels:
                for metric_label in self.external_metric_labels:
                    header_extra2 += '{label:<10s} | '.format(label=metric_label)

            header_extra2 += '{time:<15s}'.format(
                time=' '*15,
            )

            header_main = '    {epoch:<5s} | {loss:<8s} | {val_loss:<8s} | {train:<8s} | {validation:<8s} | '.format(
                epoch='Epoch',
                loss='Train',
                val_loss='Val',
                train='Train',
                validation='Val',
            )
            if self.external_metric_labels:
                for metric_label in self.external_metric_labels:
                    header_main += '{label:<10s} | '.format(label='Val')

            header_main += '{time:<15s}'.format(
                time='Time'
            )

            sep = '    {epoch:<5s} + {loss:<8s} + {val_loss:<8s} + {train:<8s} + {validation:<8s} + '.format(
                epoch='-'*5,
                loss='-'*8,
                val_loss='-' * 8,
                train='-'*8,
                validation='-'*8,
            )
            if self.external_metric_labels:
                for metric_label in self.external_metric_labels:
                    sep += '{external_value:<10s} + '.format(
                        external_value='-'*10,
                    )
            sep += '{time:<15s}'.format(
                time='-'*15,
            )
            self.logger.info(header_extra1)
            self.logger.info(header_extra2)
            self.logger.info(header_main)
            self.logger.info(sep)

        elif self.progress_bar is None:
            self.progress_bar = tqdm(total=self.epochs,
                                     initial=self.epoch,
                                     file=sys.stdout,
                                     desc='    {0:>6s}'.format('Learn'),
                                     leave=False,
                                     miniters=1,
                                     disable=self.disable_progress_bar
                                     )

    def on_train_end(self, logs=None):
        if not self.log_progress and self.close_progress_bar:
            self.progress_bar.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

        self.seen = 0
        self.timer.start()

    def on_batch_begin(self, batch, logs=None):
        if 'steps' in self.params:
            if self.seen < self.params['steps']:
                self.log_values = []
        elif 'samples' in self.params:
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
        self.timer.stop()
        self.epoch = epoch

        logs = logs or {}

        # Reset values
        self.postfix['l_tra'] = None
        self.postfix['l_val'] = None
        self.postfix['m_tra'] = None
        self.postfix['m_val'] = None

        # Collect values
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                if k == 'loss':
                    self.data['l_tra'][self.epoch] = logs[k]
                    self.postfix['l_tra'] = '{:4.3f}'.format(logs[k])

                elif k == 'val_loss':
                    self.data['l_val'][self.epoch] = logs[k]
                    self.postfix['l_val'] = '{:4.3f}'.format(logs[k])

                elif self.metric and k.endswith(self.metric):
                    if k.startswith('val_'):
                        self.data['m_val'][self.epoch] = logs[k]
                        self.postfix['m_val'] = '{:4.3f}'.format(logs[k])
                    else:
                        self.data['m_tra'][self.epoch] = logs[k]
                        self.postfix['m_tra'] = '{:4.3f}'.format(logs[k])

        for metric_label in self.external_metric_labels:
            if metric_label in self.external_metric:
                metric_name = self.external_metric_labels[metric_label]
                value = self.external_metric[metric_label]
                if metric_name.endswith('f_measure') or metric_name.endswith('f_score'):
                    self.postfix[metric_label] = '{:3.1f}'.format(value*100)
                else:
                    self.postfix[metric_label] = '{:4.3f}'.format(value)

        if (not self.manual_update or
           (self.epoch - self.last_update_epoch > 0 and (self.epoch+1) % self.manual_update_interval)):
            # Update logged progress
            if self.log_progress:
                self.update_progress_log()

        # Increase iteration count and update progress bar
        if not self.log_progress:
            self.update_progress_bar(increase=1)

    def update(self):
        """Update

        """

        if self.log_progress:
            self.update_progress_log()
        else:
            self.update_progress_bar()

        self.last_update_epoch = self.epoch

    def update_progress_log(self):
        """Update progress to logging interface

        """

        if self.log_progress and self.epoch - self.last_update_epoch:
            output = '    '
            output += '{epoch:<5s} |'.format(epoch='{:d}'.format(self.epoch))

            output += ' {loss:<8s} |'.format(loss='{:4.6f}'.format(self.data['l_tra'][self.epoch]))

            if 'l_val' in self.postfix:
                output += ' {val_loss:<8s} |'.format(val_loss='{:4.6f}'.format(self.data['l_val'][self.epoch]))
            else:
                output += ' {val_loss:<8s} |'.format(val_loss=' '*8)

            output += ' {train:<8s} |'.format(train='{:4.6f}'.format(self.data['m_tra'][self.epoch]))

            if self.postfix['m_val']:
                output += ' {validation:<8s} |'.format(validation='{:4.6f}'.format(self.data['m_val'][self.epoch]))
            else:
                output += ' {validation:<8s} |'.format(validation=' '*8)

            for metric_label in self.external_metric_labels:
                if metric_label in self.external_metric:
                    value = self.data[metric_label][self.epoch]

                    if numpy.isnan(value):
                        value = ' '*10
                    else:
                        if(self.external_metric_labels[metric_label].endswith('f_measure') or
                           self.external_metric_labels[metric_label].endswith('f_score')):
                            value = '{:3.3f}'.format(float(value)*100)
                        else:
                            value = '{:4.3f}'.format(float(value))

                    output += ' {external_value:<10s} |'.format(
                        external_value=value
                    )
                else:
                    output += ' {external_value:<10s} |'.format(
                        external_value=' '*10
                    )
            output += ' {time:<15s}'.format(
                time=self.timer.get_string()
            )
            self.logger.info(output)

    def update_progress_bar(self, increase=0):
        """Update progress to tqdm progress bar

        """

        self.progress_bar.set_postfix(self.postfix)
        self.progress_bar.update(increase)

    def add_external_metric(self, metric_id):
        """Add external metric to be monitored

        Parameters
        ----------
        metric_id : str
            Metric name

        """

        if metric_id not in self.external_metric_labels:
            self.external_metric_labels[metric_id] = metric_id

        if metric_id not in self.data:
            self.data[metric_id] = numpy.empty((self.epochs,))
            self.data[metric_id][:] = numpy.nan

    def set_external_metric_value(self, metric_label, metric_value):
        """Add external metric value

        Parameters
        ----------
        metric_label : str
            Metric label

        metric_value : numeric
            Metric value

        """

        self.external_metric[metric_label] = metric_value
        self.data[metric_label][self.epoch] = metric_value

    def close(self):
        """Manually close progress logging

        """

        if not self.log_progress and self.close_progress_bar:
            self.progress_bar.close()


class ProgressPlotterCallback(ProgressLoggerCallback):
    """Keras callback to plot progress during the training process and save final progress into figure.
    Implements Keras Callback API.

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        epochs : int
            Total amount of epochs

        metric : str
            Metric name

        manual_update : bool
            Manually update callback, use this to when injecting external metrics
            Default value True

        interactive : bool
            Show plot during the training and update with plotting rate
            Default value True

        plotting_rate : int
            Plot update rate in seconds
            Default value 10

        save : bool
            Save plot on disk, plotting rate applies

        filename : str
            Filename of figure
            Default value 1

        focus_span : int
            Epoch amount to highlight, and show separately in the plot.
            Default value 10

        """

        super(ProgressPlotterCallback, self).__init__(*args, **kwargs)
        self.filename = kwargs.get('filename')

        # Get file format for the output plot
        file_extension = os.path.splitext(self.filename)[1]
        if file_extension == '.eps':
            self.format = 'eps'
        elif file_extension == '.svg':
            self.format = 'svg'
        elif file_extension == '.pdf':
            self.format = 'pdf'
        elif file_extension == '.png':
            self.format = 'png'

        self.plotting_rate = kwargs.get('plotting_rate', 10)
        self.interactive = kwargs.get('interactive', True)
        self.save = kwargs.get('save', True)
        self.focus_span = kwargs.get('focus_span', 10)

        if self.focus_span > self.epochs:
            self.focus_span = self.epochs

        self.timer.start()
        self.data = {
            'l_tra': numpy.empty((self.epochs,)),
            'l_val': numpy.empty((self.epochs,)),
            'm_tra': numpy.empty((self.epochs,)),
            'm_val': numpy.empty((self.epochs,)),
        }
        self.data['l_tra'][:] = numpy.nan
        self.data['l_val'][:] = numpy.nan
        self.data['m_tra'][:] = numpy.nan
        self.data['m_val'][:] = numpy.nan

        for metric_label in self.external_metric_labels:
            self.data[metric_label] = numpy.empty((self.epochs,))
            self.data[metric_label][:] = numpy.nan

        self.ax1_1 = None
        self.ax1_2 = None

        self.ax2_1 = None
        self.ax2_2 = None

        self.extra_main = {}
        self.extra_highlight = {}

        import matplotlib.pyplot as plt
        import warnings
        import matplotlib.cbook
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

        figure_height = 8
        if len(self.external_metric_labels) > 2:
            figure_height = 8 + len(self.external_metric_labels)

        self.figure = plt.figure(num=None, figsize=(18, figure_height), dpi=80, facecolor='w', edgecolor='k')
        self.draw()
        if self.interactive:
            plt.show(block=False)
            plt.pause(0.1)

    def draw(self):
        """Draw plot

        """

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        plt.figure(self.figure.number)
        row_count = 2+len(self.external_metric_labels)

        self.ax1_1 = plt.subplot2grid((row_count, 4), (0, 0), rowspan=1, colspan=3)
        self.ax1_2 = plt.subplot2grid((row_count, 4), (0, 3), rowspan=1, colspan=1)

        self.ax2_1 = plt.subplot2grid((row_count, 4), (1, 0), rowspan=1, colspan=3)
        self.ax2_2 = plt.subplot2grid((row_count, 4), (1, 3), rowspan=1, colspan=1)

        self.extra_main = {}
        self.extra_highlight = {}
        row_id = 2
        for metric_label in self.external_metric_labels:
            self.extra_main[metric_label] = plt.subplot2grid((row_count, 4), (row_id, 0), rowspan=1, colspan=3)
            self.extra_highlight[metric_label] = plt.subplot2grid((row_count, 4), (row_id, 3), rowspan=1, colspan=1)
            row_id += 1

        span = [self.epoch - self.focus_span, self.epoch]
        if span[0] < 0:
            span[0] = 0

        # PLOT 1 / Main
        self.ax1_1.cla()
        self.ax1_1.set_title('Loss')
        self.ax1_1.set_ylabel('Model Loss')
        self.ax1_1.plot(
            numpy.arange(self.epochs),
            self.data['l_tra'],
            lw=3,
            color='red',
        )
        self.ax1_1.plot(
            numpy.arange(self.epochs),
            self.data['l_val'],
            lw=3,
            color='green',
        )
        self.ax1_1.add_patch(
            patches.Rectangle(
                (span[0], self.ax1_1.get_ylim()[0]),  # (x,y)
                width=span[1]-span[0],
                height=self.ax1_1.get_ylim()[1],
                facecolor="#000000",
                alpha=0.05
            )
        )
        # Horizontal lines
        if not numpy.all(numpy.isnan(self.data['l_tra'])):
            self.ax1_1.axhline(y=numpy.nanmin(self.data['l_tra']), lw=1, color='red', linestyle='--')
            self.ax1_1.axhline(y=numpy.nanmin(self.data['l_val']), lw=1, color='green', linestyle='--')

        self.ax1_1.legend(['Train', 'Validation'], loc='upper right')
        self.ax1_1.set_xlim([0, self.epochs - 1])
        self.ax1_1.set_xticklabels([])
        self.ax1_1.grid(True)

        # PLOT 1 / Highlighted area
        self.ax1_2.cla()
        self.ax1_2.set_title('Loss / Highlighted area')
        self.ax1_2.set_ylabel('Model Loss')
        self.ax1_2.plot(
            numpy.arange(span[0], span[1]),
            self.data['l_tra'][span[0]:span[1]],
            lw=3,
            color='red',
        )
        self.ax1_2.plot(
            numpy.arange(span[0], span[1]),
            self.data['l_val'][span[0]:span[1]],
            lw=3,
            color='green',
        )
        self.ax1_2.set_xticklabels([])
        self.ax1_2.grid(True)
        self.ax1_2.yaxis.tick_right()
        self.ax1_2.yaxis.set_label_position("right")

        # PLOT 2 / Main
        self.ax2_1.cla()
        self.ax2_1.set_title('Metric')
        self.ax2_1.set_ylabel(self.metric)
        # Plots
        self.ax2_1.plot(
            numpy.arange(self.epochs),
            self.data['m_tra'],
            lw=3,
            color='red',
        )
        self.ax2_1.plot(
            numpy.arange(self.epochs),
            self.data['m_val'],
            lw=3,
            color='green',
        )
        # Horizontal lines
        if not numpy.all(numpy.isnan(self.data['m_tra'])):
            if self.get_operator(metric=self.metric) == numpy.greater:
                h_tra_line_y = numpy.nanmax(self.data['m_tra'])
                h_val_line_y = numpy.nanmax(self.data['m_val'])
            else:
                h_tra_line_y = numpy.nanmin(self.data['m_tra'])
                h_val_line_y = numpy.nanmin(self.data['m_tra'])
            self.ax2_1.axhline(y=h_tra_line_y, lw=1, color='red', linestyle='--')
            self.ax2_1.axhline(y=h_val_line_y, lw=1, color='green', linestyle='--')

        self.ax2_1.add_patch(
            patches.Rectangle(
                (span[0], self.ax2_1.get_ylim()[0]),  # (x,y)
                width=span[1]-span[0],
                height=self.ax2_1.get_ylim()[1],
                facecolor="#000000",
                alpha=0.05
            )
        )

        if self.get_operator(metric=self.metric) == numpy.greater:
            legend_location = 'lower right'
        else:
            legend_location = 'upper right'

        self.ax2_1.legend(['Train', 'Validation'], loc=legend_location)
        self.ax2_1.set_xlim([0, self.epochs - 1])
        if self.external_metric_labels:
            self.ax2_1.set_xticklabels([])
        self.ax2_1.grid(True)

        # PLOT 2 / Highlighted area
        self.ax2_2.cla()
        self.ax2_2.set_title('Metric / Highlighted area')
        self.ax2_2.set_ylabel(self.metric)
        self.ax2_2.plot(
            numpy.arange(span[0], span[1]),
            self.data['m_tra'][span[0]:span[1]],
            lw=3,
            color='red',
        )
        self.ax2_2.plot(
            numpy.arange(span[0], span[1]),
            self.data['m_val'][span[0]:span[1]],
            lw=3,
            color='green',
        )
        self.ax2_2.set_xticklabels([])
        self.ax2_2.grid(True)
        self.ax2_2.yaxis.tick_right()
        self.ax2_2.yaxis.set_label_position("right")

        for mid, metric_label in enumerate(self.external_metric_labels):
            metric_name = self.external_metric_labels[metric_label]

            if metric_name.endswith('f_measure') or metric_name.endswith('f_score'):
                factor = 100
            else:
                factor = 1

            # PLOT 3 / Main
            self.extra_main[metric_label].cla()
            self.extra_main[metric_label].set_title('External metric')
            self.extra_main[metric_label].set_ylabel(str(metric_label))

            mask = numpy.isfinite(self.data[metric_label])
            self.extra_main[metric_label].plot(
                numpy.arange(self.epochs)[mask],
                self.data[metric_label][mask]*factor,
                lw=3,
                color='green',
                marker='o',
            )
            self.extra_main[metric_label].add_patch(
                patches.Rectangle(
                    (span[0], self.extra_main[metric_label].get_ylim()[0]),  # (x,y)
                    width=span[1]-span[0],
                    height=self.extra_main[metric_label].get_ylim()[1],
                    facecolor="#000000",
                    alpha=0.05
                )
            )

            # Horizontal lines
            if not numpy.all(numpy.isnan(self.data[metric_label][mask])):
                if self.get_operator(metric=str(metric_label)) == numpy.greater:
                    h_extra_line_y = numpy.nanmax((self.data[metric_label][mask]*factor))
                else:
                    h_extra_line_y = numpy.nanmin((self.data[metric_label][mask]*factor))
                self.extra_main[metric_label].axhline(y=h_extra_line_y, lw=1, color='blue', linestyle='--')

            if self.get_operator(metric=self.metric) == numpy.greater:
                legend_location = 'lower right'
            else:
                legend_location = 'upper right'

            self.extra_main[metric_label].legend(['Validation'], loc=legend_location)
            self.extra_main[metric_label].set_xlim([0, self.epochs - 1])

            if (mid + 1) < len(self.external_metric_labels):
                self.extra_main[metric_label].set_xticklabels([])
            else:
                self.extra_main[metric_label].set_xlabel('Epochs')

            self.extra_main[metric_label].grid(True)

            # PLOT 3 / Highlighted area
            self.extra_highlight[metric_label].cla()
            self.extra_highlight[metric_label].set_title('External metric / Highlighted area')
            self.extra_highlight[metric_label].set_ylabel(str(metric_label))
            highlight_data = self.data[metric_label][span[0]:span[1]]*factor
            mask = numpy.isfinite(highlight_data)
            self.extra_highlight[metric_label].plot(
                numpy.arange(span[0], span[1])[mask],
                highlight_data[mask],
                lw=3,
                color='green',
                marker='o',
            )

            if (mid + 1) < len(self.external_metric_labels):
                self.extra_highlight[metric_label].set_xticklabels([])
            else:
                self.extra_highlight[metric_label].set_xlabel('Epochs')

            self.extra_highlight[metric_label].yaxis.tick_right()
            self.extra_highlight[metric_label].yaxis.set_label_position("right")
            self.extra_highlight[metric_label].grid(True)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.02, hspace=0.2)

    def on_train_begin(self, logs=None):
        if self.epochs is None:
            self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

        logs = logs or {}

        # Collect values
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                if k == 'loss':
                    self.data['l_tra'][self.epoch] = logs[k]

                elif k == 'val_loss':
                    self.data['l_val'][self.epoch] = logs[k]

                elif self.metric and k.endswith(self.metric):
                    if k.startswith('val_'):
                        self.data['m_val'][self.epoch] = logs[k]
                    else:
                        self.data['m_tra'][self.epoch] = logs[k]

        if not self.manual_update:
            # Update logged progress
            self.update()

    def update(self):
        """Update

        """

        import matplotlib.pyplot as plt
        if self.timer.elapsed() > self.plotting_rate:
            self.draw()
            self.figure.canvas.flush_events()
            if self.interactive:
                plt.pause(0.01)

            if self.save:
                plt.savefig(self.filename, bbox_inches='tight', format=self.format, dpi=1000)

            self.timer.start()

    def add_external_metric(self, metric_label):
        """Add external metric to be monitored

        Parameters
        ----------
        metric_label : str
            Metric label

        """

        if metric_label not in self.external_metric_labels:
            self.external_metric_labels[metric_label] = metric_label

        if metric_label not in self.data:
            self.data[metric_label] = numpy.empty((self.epochs,))
            self.data[metric_label][:] = numpy.nan

    def set_external_metric_value(self, metric_label, metric_value):
        """Add external metric value

        Parameters
        ----------
        metric_label : str
            Metric label

        metric_value : numeric
            Metric value

        """

        self.external_metric[metric_label] = metric_value
        self.data[metric_label][self.epoch] = metric_value

    def close(self):
        """Manually close progress logging

        """

        import matplotlib.pyplot as plt

        if self.save:
            self.draw()
            plt.savefig(self.filename, bbox_inches='tight', format=self.format, dpi=1000)

        plt.close(self.figure)


class StopperCallback(BaseCallback):
    """Keras callback to stop training when improvement has not seen in specified amount of epochs.
    Implements Keras Callback API.

    Callback is very similar to standard ``EarlyStopping`` Keras callback, however it adds support for external metrics
    (calculated outside Keras training process).

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        epochs : int
            Total amount of epochs

        manual_update : bool
            Manually update callback, use this to when injecting external metrics
            Default value True

        monitor : str
            Metric value to be monitored
            Default value "val_loss"

        patience : int
            Number of epochs with no improvement after which training will be stopped.
            Default value 0

        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
            Default value 0

        initial_delay : int
            Amount of epochs to wait at the beginning before quantity is monitored.
            Default value 10

        """

        super(StopperCallback, self).__init__(*args, **kwargs)

        self.monitor = kwargs.get('monitor', 'val_loss')
        self.patience = kwargs.get('patience', 0)
        self.min_delta = kwargs.get('min_delta', 0)
        self.initial_delay = kwargs.get('initial_delay', 10)

        self.wait = None
        self.stopped_epoch = None
        self.model = None
        self.params = None
        self.last_update_epoch = 0
        self.stopped = False
        self.logger = logging.getLogger(__name__)

        self.metric_data = {
            self.monitor: numpy.empty((self.epochs,))
        }
        self.metric_data[self.monitor][:] = numpy.nan

        mode = kwargs.get('mode', 'auto')
        if mode not in ['min', 'max', 'auto']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = numpy.less
        elif mode == 'max':
            self.monitor_op = numpy.greater
        else:
            self.monitor_op = self.get_operator(metric=self.monitor)

        self.best = numpy.Inf if self.monitor_op == numpy.less else -numpy.Inf

        if self.monitor_op == numpy.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        if self.epochs is None:
            self.epochs = self.params['epochs']

        if self.wait is None:
            self.wait = 0

        if self.stopped_epoch is None:
            self.stopped_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.monitor in logs:
            self.metric_data[self.monitor][self.epoch] = logs.get(self.monitor)

        if not self.manual_update:
            self.update()

    def set_external_metric_value(self, metric_label, metric_value):
        """Add external metric value

        Parameters
        ----------
        metric_label : str
            Metric label

        metric_value : numeric
            Metric value

        """

        if metric_label not in self.metric_data:
            self.metric_data[metric_label] = numpy.empty((self.epochs,))
            self.metric_data[metric_label][:] = numpy.nan

        self.metric_data[metric_label][self.epoch] = metric_value

    def stop(self):
        return self.stopped

    def update(self):
        if self.epoch > self.initial_delay:
            # get current metric value
            current = self.metric_data[self.monitor][self.epoch]
            if numpy.isnan(current):
                message = '{name}: Metric to monitor is Nan, metric:[{metric}]'.format(
                    name=self.__class__.__name__,
                    metric=self.monitor
                )
                self.logger.exception(message)
                raise ValueError(message)

            if self.monitor_op(current - self.min_delta, self.best):
                # New best value found
                self.best = current
                self.wait = 0

            else:
                if self.wait >= self.patience:
                    # Stopping criteria met => return false
                    self.stopped_epoch = self.epoch
                    self.model.stop_training = True
                    self.logger.info('  Stopping criteria met at epoch[{epoch:d}]'.format(
                        epoch=self.epoch,
                    ))
                    self.logger.info('    metric[{metric}], patience[{patience:d}]'.format(
                        metric=self.monitor,
                        current='{:4.4f}'.format(current),
                        patience=self.patience
                    ))
                    self.logger.info('  ')
                    self.stopped = True
                    return self.stopped

                # Increase waiting counter
                self.wait += self.epoch - self.last_update_epoch

        self.last_update_epoch = self.epoch
        return self.stopped


class StasherCallback(BaseCallback):
    """Keras callback to monitor training process and store best model. Implements Keras Callback API.

    This callback is very similar to standard ``ModelCheckpoint`` Keras callback, however it adds support for external
    metrics (metrics calculated outside Keras training process).

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        epochs : int
            Total amount of epochs

        manual_update : bool
            Manually update callback, use this to when injecting external metrics
            Default value True

        monitor : str
            Metric to monitor
            Default value 'val_loss'

        mode : str
            Which way metric is interpreted, values {auto, min, max}
            Default value 'auto'

        period : int
            Disable tqdm based progress bar
            Default value 1

        initial_delay : int
            Amount of epochs to wait at the beginning before quantity is monitored.
            Default value 10

        save_weights : bool
            Save weight to the disk
            Default value False

        file_path : str
            File name for model weight
            Default value None
        """

        super(StasherCallback, self).__init__(*args, **kwargs)

        self.monitor = kwargs.get('monitor', 'val_loss')
        self.period = kwargs.get('period', 1)
        self.initial_delay = kwargs.get('initial_delay', 10)

        self.save_weights = kwargs.get('save_weights', False)
        self.file_path = kwargs.get('file_path', None)

        self.epochs_since_last_save = 0
        self.logger = logging.getLogger(__name__)

        mode = kwargs.get('mode', 'auto')
        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = numpy.less

        elif mode == 'max':
            self.monitor_op = numpy.greater

        else:
            self.monitor_op = self.get_operator(metric=self.monitor)


        self.best = numpy.Inf if self.monitor_op == numpy.less else -numpy.Inf

        self.metric_data = {
            self.monitor: numpy.empty((self.epochs,))
        }
        self.metric_data[self.monitor][:] = numpy.nan
        self.best_model_weights = None
        self.best_model_epoch = 0
        self.last_logs = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.monitor in logs:
            self.metric_data[self.monitor][self.epoch] = logs.get(self.monitor)

        self.last_logs = logs

        if not self.manual_update:
            self.update()

    def update(self):
        if self.epoch > self.initial_delay:
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0

                current = self.metric_data[self.monitor][self.epoch]
                if numpy.isnan(current):
                    message = '{name}: Metric to monitor is Nan, metric:[{metric}]'.format(
                        name=self.__class__.__name__,
                        metric=self.monitor
                    )
                    self.logger.exception(message)
                    raise ValueError(message)

                else:
                    if self.monitor_op(current, self.best):

                        # Store the best
                        self.best = current
                        self.best_model_weights = self.model.get_weights()
                        self.best_model_epoch = self.epoch

                        if self.save_weights and self.file_path:
                            # Save weight on disk
                            logs = self.last_logs
                            if self.monitor not in logs:
                                logs[self.monitor] = current

                            file_path = self.file_path.format(epoch=self.epoch, **self.last_logs)
                            self.model.save_weights(file_path, overwrite=True)

    def set_external_metric_value(self, metric_label, metric_value):
        """Add external metric value

        Parameters
        ----------
        metric_label : str
            Metric label

        metric_value : numeric
            Metric value

        """

        if metric_label not in self.metric_data:
            self.metric_data[metric_label] = numpy.empty((self.epochs,))
            self.metric_data[metric_label][:] = numpy.nan
        self.metric_data[metric_label][self.epoch] = metric_value

    def get_best(self):
        """Return best model seen

        Returns
        -------
        dict
            Dictionary with keys 'weights', 'epoch', 'metric_value'
        """

        return {
            'epoch': self.best_model_epoch,
            'weights': self.best_model_weights,
            'metric_value': self.best,
        }

    def log(self):
        """Print information about the best model into logging interface
        """

        self.logger.info('  Best model weights at epoch[{epoch:d}]'.format(epoch=self.best_model_epoch))
        self.logger.info('    metric[{metric}]={best}'.format(
                metric=self.monitor,
                best='{:4.4f}'.format(self.best)
            )
        )
        self.logger.info('  ')


class BaseDataGenerator(object):
    """Base class for data generator.
    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        files : list

        data_filenames : dict

        annotations : dict

        class_labels : list of str

        hop_length_seconds : float
            Default value 0.2

        shuffle : bool
            Default value True

        batch_size : int
            Default value 64

        buffer_size : int
            Default value 256

        """

        self.method = 'base_generator'

        # Data
        self.item_list = copy.copy(kwargs.get('files', []))
        self.data_filenames = kwargs.get('data_filenames', {})
        self.annotations = kwargs.get('annotations', {})

        # Activity matrix
        self.class_labels = kwargs.get('class_labels', [])
        self.hop_length_seconds = kwargs.get('hop_length_seconds', 0.2)

        self.shuffle = kwargs.get('shuffle', True)
        self.batch_size = kwargs.get('batch_size', 64)
        self.buffer_size = kwargs.get('buffer_size', 256)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        # Internal state variables
        self.batch_index = 0
        self.item_index = 0
        self.data_position = 0

        # Initialize data buffer
        self.data_buffer = DataBuffer(size=self.buffer_size)

        if self.buffer_size >= len(self.item_list):
            # Fill data buffer at initialization if it fits fully to the buffer

            for current_item in self.item_list:
                self.process_item(item=current_item)
                if self.data_buffer.full():
                    break

        self._data_size = None
        self._input_size = None

    @property
    def steps_count(self):
        """Number of batches in one epoch
        """

        num_batches = int(numpy.ceil(self.data_size / float(self.batch_size)))

        if num_batches > 0:
            return num_batches
        else:
            return 1

    @property
    def input_size(self):
        """Length of input feature vector
        """
        if self._input_size is None:
            # Load first item
            first_item = list(self.data_filenames.keys())[0]
            self.process_item(item=first_item)

            # Get Feature vector length
            self._input_size = self.data_buffer.get(key=first_item)[0].shape[-1]

        return self._input_size

    @property
    def data_size(self):
        """Total data amount
        """
        if self._data_size is None:
            self._data_size = 0
            for current_item in self.item_list:
                self.process_item(item=current_item)
                data, meta = self.data_buffer.get(key=current_item)

                # Accumulate feature matrix length
                self._data_size += data.shape[0]

        return self._data_size

    def info(self):
        """Information logging
        """
        info = [
            '  Generator',
            '    Shuffle \t[{shuffle}]'.format(shuffle='True' if self.shuffle else 'False'),
            '    Epoch size\t[{steps:d} batches]'.format(steps=self.steps_count),
            '    Buffer size \t[{buffer_size:d} files]'.format(buffer_size=self.buffer_size),
            ' '
        ]
        return info

    def process_item(self, item):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass


class FeatureGenerator(BaseDataGenerator):
    """Feature data generator
    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        files : list of str
            List of active item identifies, usually filenames

        data_filenames : dict of dicts
            Data structure keyed with item identifiers (defined with files parameter), data dict feature extractor
            labels as keys and values the filename on disk.

        annotations : dict of MetaDataContainers or MetaDataItems
            Annotations for all items keyed with item identifiers

        class_labels : list of str
            Class labels in a list

        hop_length_seconds : float
            Analysis frame hop length in seconds
            Default value 0.2

        shuffle : bool
            Shuffle data before each epoch
            Default value True

        batch_size : int
            Batch size to generate
            Default value 64

        buffer_size : int
            Internal item buffer size, set large enough for smaller dataset to avoid loading
            Default value 256

        data_processor : class
            Data processor class used to process load features

        data_refresh_on_each_epoch : bool
            Internal data buffer reset at the start of each epoch
            Default value False

        label_mode : str ('event', 'scene')
            Activity matrix forming mode.
            Default value "event"

        """

        self.data_processor = kwargs.get('data_processor')
        self.data_refresh_on_each_epoch = kwargs.get('data_refresh_on_each_epoch', False)
        self.label_mode = kwargs.get('label_mode', 'event')

        super(FeatureGenerator, self).__init__(*args, **kwargs)

        self.method = 'feature'

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        if self.label_mode not in ['event', 'scene']:
            message = '{name}: Label mode unknown [{label_mode}]'.format(
                name=self.__class__.__name__,
                metric=self.label_mode
            )
            self.logger.exception(message)
            raise ValueError(message)

    def process_item(self, item):
        if not self.data_buffer.key_exists(key=item):
            current_data, current_length = self.data_processor.load(
                feature_filename_dict=self.data_filenames[item]
            )
            current_activity_matrix = self.get_activity_matrix(
                annotation=self.annotations[item],
                data_length=current_length
            )
            self.data_buffer.set(key=item, data=current_data, meta=current_activity_matrix)

    def on_epoch_start(self):
        self.batch_index = 0
        if self.shuffle:
            # Shuffle item list order
            numpy.random.shuffle(self.item_list)

        if self.data_refresh_on_each_epoch:
            # Force reload of data
            self.data_buffer.clear()

    def generator(self):
        """Generator method

        Returns
        -------
        ndarray
            data batches
        """

        while True:
            # Start of epoch
            self.on_epoch_start()

            batch_buffer_data = []
            batch_buffer_meta = []

            # Go through items
            for item in self.item_list:
                # Load item data into buffer
                self.process_item(item=item)

                # Fetch item from buffer
                data, meta = self.data_buffer.get(key=item)

                # Data indexing
                data_ids = numpy.arange(data.shape[0])

                # Shuffle data order
                if self.shuffle:
                    numpy.random.shuffle(data_ids)

                for data_id in data_ids:
                    if len(batch_buffer_data) == self.batch_size:
                        # Batch buffer full, yield data
                        yield (
                            numpy.concatenate(
                                numpy.expand_dims(batch_buffer_data, axis=0)
                            ),
                            numpy.concatenate(
                                numpy.expand_dims(batch_buffer_meta, axis=0)
                            )
                        )

                        # Empty batch buffers
                        batch_buffer_data = []
                        batch_buffer_meta = []

                        # Increase batch counter
                        self.batch_index += 1

                    # Collect data fro the batch
                    batch_buffer_data.append(data[data_id])
                    batch_buffer_meta.append(meta[data_id])

            if len(batch_buffer_data):
                # Last batch, usually not full
                yield (
                    numpy.concatenate(
                        numpy.expand_dims(batch_buffer_data, axis=0)
                    ),
                    numpy.concatenate(
                        numpy.expand_dims(batch_buffer_meta, axis=0)
                    ),
                )
                # Increase batch counter
                self.batch_index += 1

            # End of epoch
            self.on_epoch_end()

    def get_activity_matrix(self, annotation, data_length):
        """Convert annotation into activity matrix and run it through data processor.
        """

        event_roll = None
        if self.label_mode == 'event':
            # Event activity, event onset and offset specified
            event_roll = EventRoll(metadata_container=annotation,
                                   label_list=self.class_labels,
                                   time_resolution=self.hop_length_seconds
                                   )
            event_roll = event_roll.pad(length=data_length)

        elif self.label_mode == 'scene':
            # Scene activity, one-hot activity throughout whole file
            pos = self.class_labels.index(annotation.scene_label)
            event_roll = numpy.zeros((data_length, len(self.class_labels)))
            event_roll[:, pos] = 1

        if event_roll is not None:
            return self.data_processor.process_activity_data(
                activity_data=event_roll
            )

        else:
            return None
