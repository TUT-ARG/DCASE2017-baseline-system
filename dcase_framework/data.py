#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data utils
==========

Utility classes related to data handling.

DataSequencer
^^^^^^^^^^^^^

Data sequencer class to process data matrix into sequences (images). Sequences can overlap. Sequencing grid can be
altered between calls.

.. autosummary::
    :toctree: generated/

    DataSequencer
    DataSequencer.process
    DataSequencer.increase_shifting

DataProcessor
^^^^^^^^^^^^^

Data processor class to process raw features into data suitable for machine learning algorithms. Feature processing
chain and data processing chain are defined during the class construction, and these processing chains are applied
to the input data.

.. autosummary::
    :toctree: generated/

    DataProcessor
    DataProcessor.load
    DataProcessor.process
    DataProcessor.process_features
    DataProcessor.process_activity_data
    DataProcessor.process_data
    DataProcessor.call_method

DataBuffer
^^^^^^^^^^

Data buffering class, which can be used to store data and meta data associated to the item. Item data is accessed
through item key. When internal buffer is filled, oldest item is replaced.

.. autosummary::
    :toctree: generated/

    DataBuffer
    DataBuffer.count
    DataBuffer.full
    DataBuffer.key_exists
    DataBuffer.set
    DataBuffer.get
    DataBuffer.clear

ProcessingChain
^^^^^^^^^^^^^^^

Data processing chain class, inherited from list.

.. autosummary::
    :toctree: generated/

    ProcessingChain
    ProcessingChain.process
    ProcessingChain.call_method

"""

from __future__ import print_function, absolute_import
from six import iteritems
import logging
import numpy
import copy
import collections
from .features import FeatureRepository


class DataBuffer(object):
    """Data buffer (FIFO)

    Buffer can store data and meta data associated to it.

    """

    def __init__(self, *args, **kwargs):
        """__init__ method.

        Parameters
        ----------
        size : int
            Number of item to store in the buffer
            Default value 10

        """

        self.size = kwargs.get('size', 10)

        self.index = collections.deque(maxlen=self.size)
        self.data_buffer = collections.deque(maxlen=self.size)
        self.meta_buffer = collections.deque(maxlen=self.size)

    def count(self):
        """Buffer usage

        Returns
        -------
        buffer length: int

        """

        return len(self.index)

    def full(self):
        """Buffer full

        Returns
        -------
        bool

        """

        if self.count() == self.size:
            return True
        else:
            return False

    def key_exists(self, key):
        """Check that key exists in the buffer

        Parameters
        ----------
        key : str or number
            Key value

        Returns
        -------
        bool

        """

        if key in self.index:
            return True
        else:
            return False

    def set(self, key, data=None, meta=None):
        """Insert item to the buffer

        Parameters
        ----------
        key : str or number
            Key value
        data :
            Item data
        meta :
            Item meta

        Returns
        -------
        DataBuffer object

        """

        if not self.key_exists(key):
            self.index.append(key)
            self.data_buffer.append(data)
            self.meta_buffer.append(meta)
        return self

    def get(self, key):
        """Get item based on key

        Parameters
        ----------
        key : str or number
            Key value

        Returns
        -------
        data : (data, meta)

        """

        if self.key_exists(key):
            index = list(self.index).index(key)
            return self.data_buffer[index], self.meta_buffer[index]
        else:
            return None, None

    def clear(self):
        """Empty the buffer
        """

        self.index.clear()
        self.data_buffer.clear()
        self.meta_buffer.clear()


class DataProcessingUnitMixin(object):
    """Data processing chain unit mixin"""
    def process(self, data):
        pass


class DataSequencer(DataProcessingUnitMixin):
    """Data sequencer"""
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """__init__ method.

        Parameters
        ----------
        frames : int
            Sequence length
            Default value 10
        hop : int
            Hop value of when forming the sequence
            Default value = frames
        padding: bool
            Replicate data when sequence is not full
            Default value False
        shift_step : int
            Sequence start temporal shifting amount, is added once method increase_shifting is called
            Default value 0
        shift_border : string, {'roll', 'shift'}
            Sequence border handling when doing temporal shifting.
            Default value 'roll'
        shift_max : int
            Maximum value for temporal shift
            Default value None

        """

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        self.frames = kwargs.get('frames', 10)
        self.hop_size = kwargs.get('hop')
        if self.hop_size is None:
            self.hop_size = self.frames

        self.padding = kwargs.get('padding', False)
        self.shift = 0

        self.shift_step = kwargs.get('shift_step', 0)
        self.shift_border = kwargs.get('shift_border')

        if self.shift_border is None:
            self.shift_border = 'roll'

        if self.shift_border not in ['roll', 'shift']:
            message = '{name}: Unknown temporal shifting border handling [{border_mode}]'.format(
                name=self.__class__.__name__,
                border_mode=self.shift_border
            )
            self.logger.exception(message)
            raise IOError(message)

        self.shift_max = kwargs.get('shift_max')

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'frames': self.frames,
            'hop_size': self.hop_size,
            'padding': self.padding,
            'shift': self.shift,
            'shift_step': self.shift_step,
            'shift_border': self.shift_border,
            'shift_max': self.shift_max,
        }

    def __setstate__(self, d):
        self.frames = d['frames']
        self.hop_size = d['hop_size']
        self.padding = d['padding']
        self.shift = d['shift']
        self.shift_step = d['shift_step']
        self.shift_border = d['shift_border']
        self.shift_max = d['shift_max']
        self.logger = logging.getLogger(__name__)

    def process(self, data):
        """Process

        Parameters
        ----------
        data : numpy.ndarray
            Data

        Returns
        -------
        numpy.ndarray

        """

        # Make copy of the data to prevent modifications to the original data
        data = copy.deepcopy(data)

        # Not the most efficient way as numpy stride_tricks would produce
        # faster code, however, opted for cleaner presentation this time.
        data_length = data.shape[0]
        X = []

        if self.shift_border == 'shift':
            segment_indexes = numpy.arange(self.shift, data_length, self.hop_size)

        elif self.shift_border == 'roll':
            segment_indexes = numpy.arange(0, data_length, self.hop_size)

            if self.shift:
                # Roll data
                data = numpy.roll(
                    data,
                    shift=-self.shift,
                    axis=0
                )

        if self.padding:
            if len(segment_indexes) == 0:
                # Have at least one segment
                segment_indexes = numpy.array([0])
        else:
            # Remove segments which are not full
            segment_indexes = segment_indexes[(segment_indexes+self.frames-1) < data_length]

        for segment_start_frame in segment_indexes:
            segment_end_frame = segment_start_frame + self.frames

            frame_ids = numpy.array(range(segment_start_frame, segment_end_frame))

            if self.padding:
                # If start of matrix, pad with first frame
                frame_ids[frame_ids < 0] = 0

                # If end of the matrix, pad with last frame
                frame_ids[frame_ids > data_length - 1] = data_length - 1

            X.append(numpy.expand_dims(data[frame_ids, :], axis=0))

        if len(X) == 0:
            message = '{name}: Cannot create valid segment, adjust segment length and hop size, or use ' \
                      'padding flag.'.format(name=self.__class__.__name__)

            self.logger.exception(message)
            raise IOError(message)

        return numpy.concatenate(X)

    def increase_shifting(self, shift_step=None):
        """Increase temporal shifting

        Parameters
        ----------
        shift_step : int
            Optional value, if none given shift_step parameter given for init is used.
            Default value None

        """

        if shift_step is None:
            shift_step = self.shift_step
        self.shift += shift_step

        if self.shift_max and self.shift > self.shift_max:
            self.shift = 0


class DataProcessor(object):
    """Data processors with feature and data processing chains

    Feature processing chain comprehend all processing done to get feature matrix synchronized with meta data.

    Data processing chain is applied to the feature matrix and meta data to reshape data for machine learning.

    """
    __version__ = '0.0.1'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        feature_processing_chain : ProcessingChain
            List of processing functions

        data_processing_chain : ProcessingChain
            List of data processing functions

        """

        self.feature_processing_chain = kwargs.get('feature_processing_chain', ProcessingChain())
        self.data_processing_chain = kwargs.get('data_processing_chain', ProcessingChain())

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

    def __getstate__(self):
        # Return only needed data for pickle
        return {
            'feature_processing_chain': self.feature_processing_chain,
            'data_processing_chain': self.data_processing_chain,
        }

    def __setstate__(self, d):
        self.feature_processing_chain = d['feature_processing_chain']
        self.data_processing_chain = d['data_processing_chain']
        self.logger = logging.getLogger(__name__)

    def load(self, feature_filename_dict, process_features=True, process_data=True):
        """Load feature item

        Parameters
        ----------
        feature_filename_dict : dict of filenames
            Dict with feature extraction methods as keys and value corresponding feature file
        process_features : bool
            Apply feature processing chain.
            Default value True
        process_data : bool
            Apply data processing chain.
            Default value True
        Returns
        -------
        Processed feature data

        """

        # Load item
        feature_data = FeatureRepository(filename_dict=feature_filename_dict)

        # Process item
        return self.process(
            feature_data=feature_data,
            process_features=process_features,
            process_data=process_data
        )

    def process(self, feature_data, process_features=True, process_data=True):
        """Process feature data

        Parameters
        ----------
        feature_data : FeatureContainer
            Feature data.
        process_features : bool
            Apply feature processing chain.
            Default value True
        process_data : bool
            Apply data processing chain.
            Default value True

        Returns
        -------
        feature_data : ndarray
            Processed feature data
        feature_vector_count : int
            Number of feature vectors before data processing

        """

        # Go through the feature processing chain
        if process_features:
            feature_data = self.process_features(feature_data=feature_data)

        # Save feature matrix length before doing data processing chain
        feature_vector_count = feature_data.shape[0]

        if self.data_processing_chain and process_data:
            return self.process_data(data=feature_data.feat[0]), feature_vector_count

        else:
            return feature_data.feat[0], feature_vector_count

    def process_features(self, feature_data):
        """Process feature data

        Parameters
        ----------
        feature_data : FeatureContainer
            Feature data.

        Returns
        -------
        feature_data : ndarray
            Processed feature data

        """

        if hasattr(feature_data, 'feat'):
            feature_data = feature_data.feat[0]

        # Do processing
        feature_data = self.feature_processing_chain.process(feature_data)

        return feature_data

    def process_activity_data(self, activity_data):
        """Process activity data

        Parameters
        ----------
        activity_data : ndarray
            Activity data, usually binary matrix

        Returns
        -------
        activity_data : ndarray
            Processed activity data

        """

        return self.process_data(data=activity_data, metadata=True)

    def process_data(self, data, metadata=False):
        """Process data

        Parameters
        ----------
        data : ndarray
            Data
        metadata : bool
            Processing metadata, extra dimension added non-metadata
            Default value True

        Returns
        -------
        data : ndarray
            Processed data

        """

        if hasattr(data, 'feat'):
            data = data.feat[0]

        # Go through the data processing chain
        if self.data_processing_chain:
            data = self.data_processing_chain.process(data)

            if not metadata:
                data = numpy.expand_dims(data, axis=1)

        return data

    def call_method(self, method_name, parameters=None):
        """Call class method in the processing chain items

        Processing chain (feature and data) is gone through and given method is
        called to processing items having such method.

        Parameters
        ----------
        method_name : str
            Method name to call
        parameters : dict
            Parameters for the method
            Default value {}

        """

        self.feature_processing_chain.call_method(
            method_name=method_name,
            parameters=parameters
        )

        self.data_processing_chain.call_method(
            method_name=method_name,
            parameters=parameters
        )


class ProcessingChain(list):
    def process(self, data):
        """Process the data with processing chain

        Parameters
        ----------
        data : FeatureContainer or numpy.ndarray
            Data

        Returns
        -------
        data : numpy.ndarray
            Processed data

        """

        for step in self:
            if step is not None and hasattr(step, 'process'):
                data = step.process(data)

        return data

    def call_method(self, method_name, parameters=None):
        """Call class method in the processing chain items

        Processing chain is gone through and given method is
        called to processing items having such method.

        Parameters
        ----------
        method_name : str
            Method name to call
        parameters : dict
            Parameters for the method
            Default value {}

        """

        parameters = parameters or {}

        for item in self:
            if hasattr(item, method_name):
                getattr(item, method_name)(**parameters)
