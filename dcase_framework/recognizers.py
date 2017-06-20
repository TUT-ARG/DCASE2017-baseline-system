#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recognizers
===========
Classes for handling the recognition process.

SceneRecognizer
...............

.. autosummary::
    :toctree: generated/

    SceneRecognizer
    SceneRecognizer.process


EventRecognizer
...............

.. autosummary::
    :toctree: generated/

    EventRecognizer
    EventRecognizer.process
    EventRecognizer.process_ratio

BaseRecognizer
..............

.. autosummary::
    :toctree: generated/

    BaseRecognizer
    BaseRecognizer.collapse_probabilities
    BaseRecognizer.collapse_probabilities_windowed
    BaseRecognizer.find_contiguous_regions
    BaseRecognizer.process_activity

"""


from __future__ import print_function, absolute_import
from six import iteritems

import numpy
import logging
import copy
import scipy

from .containers import DottedDict
from .metadata import MetaDataContainer, MetaDataItem


class BaseRecognizer(object):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        params : dict
            Processing parameters

        class_labels : list of str
            Class labels in a list

        """

        self.params = DottedDict(kwargs.get('params', {}))
        self.class_labels = kwargs.get('class_labels', [])
        self.logger = kwargs.get('logger', logging.getLogger(__name__))

    def collapse_probabilities(self, probabilities, operator='sum'):
        """Collapse probabilities

        Parameters
        ----------
        probabilities : ndarray
            Probabilities to be accumulated

        operator : str ('sum', 'prod', 'mean')
            Operator to be used
            Default value "sum"

        Returns
        -------
        ndarray
            collapsed probabilities

        """

        accumulated = numpy.ones(len(self.class_labels)) * -numpy.inf
        for row_id in range(0, probabilities.shape[0]):
            if operator == 'sum':
                accumulated[row_id] = numpy.sum(probabilities[row_id, :])
            elif operator == 'prod':
                accumulated[row_id] = numpy.prod(probabilities[row_id, :])
            elif operator == 'mean':
                accumulated[row_id] = numpy.mean(probabilities[row_id, :])
            else:
                message = '{name}: Unknown accumulation type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=operator
                )

                self.logger.exception(message)
                raise AssertionError(message)

        return accumulated

    def collapse_probabilities_windowed(self, probabilities, window_length, operator='sliding_sum'):
        """Collapse probabilities in windows

        Parameters
        ----------
        probabilities : ndarray
            Probabilities to be accumulated

        window_length : int
            Window length in analysis frame amount

        operator : str ('sliding_sum', 'sliding_mean', 'sliding_median')
            Operator to be used
            Default value "sliding_sum"

        Returns
        -------
        ndarray
            collapsed probabilities

        """

        # Lets keep the system causal and use look-back while smoothing (accumulating) likelihoods
        output_probabilities = copy.deepcopy(probabilities)
        for stop_id in range(0, probabilities.shape[0]):
            start_id = stop_id - window_length
            if start_id < 0:
                start_id = 0
            if start_id != stop_id:
                if operator == 'sliding_sum':
                    output_probabilities[start_id] = numpy.sum(probabilities[start_id:stop_id])
                elif operator == 'sliding_mean':
                    output_probabilities[start_id] = numpy.mean(probabilities[start_id:stop_id])
                elif operator == 'sliding_median':
                    output_probabilities[start_id] = numpy.median(probabilities[start_id:stop_id])
                else:
                    message = '{name}: Unknown slide and accumulate type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=operator
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                output_probabilities[start_id] = probabilities[start_id]

        return output_probabilities

    def find_contiguous_regions(self, activity_array):
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

    def process_activity(self, activity_array, window_length, operator="median_filtering"):
        """Process activity array (binary)

        Parameters
        ----------
        activity_array : ndarray
            Activity array

        window_length : int
            Window length in analysis frame amount

        operator : str ('median_filtering')
            Operator to be used
            Default value "median_filtering"

        Returns
        -------
        ndarray
            Processed activity

        """

        if operator == 'median_filtering':
            return scipy.signal.medfilt(volume=activity_array, kernel_size=window_length)
        else:
            message = '{name}: Unknown activity processing type [{type}].'.format(
                name=self.__class__.__name__,
                type=operator
            )

            self.logger.exception(message)
            raise AssertionError(message)


class SceneRecognizer(BaseRecognizer):
    """Multi-class single label recognition

    **Parameters**

    +--------------------------------+--------------------+------------------------------------------------------------+
    | Field name                     | Value type         | Description                                                |
    +================================+====================+============================================================+
    | **frame_accumulation**, Defining frame probability accumulation.                                                 |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | enable                         | bool               | Enable frame probability accumulation.                     |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | type                           | string             | Operator type used to accumulate.                          |
    |                                | {sliding_sum |     |                                                            |
    |                                | sliding_mean |     |                                                            |
    |                                | sliding_median }   |                                                            |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | window_length_seconds          | float              | Window length in seconds for sliding accumulation.         |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | **frame_binarization**, Defining frame probability binarization.                                                 |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | enable                         | bool               | Enable frame probability binarization.                     |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | type                           | string             | Type of binarization:                                      |
    |                                | {frame_max |       |                                                            |
    |                                | global_threshold } | - ``frame_max``, each frame is treated individually,       |
    |                                |                    |   max of each frame is set to one, all others to zero.     |
    |                                |                    | - ``global_threshold``, global threshold, all values over  |
    |                                |                    |   the threshold are set to one.                            |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | threshold                      | float              | Threshold value. Set to null if not used.                  |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | **event_activity_processing**, Event activity processing per frame.                                              |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | enable                         | bool               | Enable activity processing.                                |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | type                           | string             | Type of decision:                                          |
    |                                | {median_filtering} |                                                            |
    |                                |                    | - ``median_filtering``, median filtering of decision       |
    |                                |                    |    inside window.                                          |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | window_length_seconds          | float              | Length of sliding window in seconds for activity           |
    |                                |                    | processing.                                                |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | **event_post_processing**, Event post processing per event.                                                      |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | enable                         | bool               | Enable event processing.                                   |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | minimum_event_length_seconds   | float              | Minimum allowed event length. Shorter events will be       |
    |                                |                    | removed.                                                   |
    +--------------------------------+--------------------+------------------------------------------------------------+
    | minimum_event_gap_second       | float              | Minimum allowed gap between events. Smaller gaps between   |
    |                                |                    | events will cause events to be merged together.            |
    +--------------------------------+--------------------+------------------------------------------------------------+


    """
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        params : dict
            Processing parameters
        class_labels : list of str
            Class labels in a list

        """

        super(SceneRecognizer, self).__init__(*args, **kwargs)
        self.method = 'scene'
        self.logger = kwargs.get('logger', logging.getLogger(__name__))

    def process(self, frame_probabilities):
        """Multi-class single label recognition.

        Parameters
        ----------
        frame_probabilities : numpy.ndarray

        Returns
        -------
        results : str
            class label

        """

        # Accumulate probabilities
        if self.params.get_path('frame_accumulation.enable', True):
            probabilities = self.collapse_probabilities(
                probabilities=frame_probabilities,
                operator=self.params.get_path('frame_accumulation.type')
            )

        else:
            # Pass probabilities
            probabilities = frame_probabilities

        # Probability binarization
        if self.params.get_path('frame_binarization.enable', True):
            if self.params.get_path('frame_binarization.type') == 'global_threshold':
                frame_decisions = numpy.argmax(
                    probabilities > self.params.get_path('frame_binarization.threshold', 0.5),
                    axis=0
                )

            elif self.params.get_path('frame_binarization.type') == 'frame_max':
                frame_decisions = numpy.argmax(probabilities, axis=0)

            else:
                message = '{name}: Unknown frame_binarization type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=self.params.get_path('frame_binarization.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        # Decision making
        classification_result_id = None
        if self.params.get_path('decision_making.enable', True):
            if self.params.get_path('decision_making.type') == 'maximum':
                classification_result_id = numpy.argmax(probabilities)

            elif self.params.get_path('decision_making.type') == 'majority_vote':
                counts = numpy.bincount(frame_decisions)
                classification_result_id = numpy.argmax(counts)

            else:
                message = '{name}: Unknown decision_making type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=self.params.get_path('decision_making.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        if classification_result_id is not None:
            if classification_result_id < len(self.class_labels):
                return self.class_labels[classification_result_id]
            else:
                return None
        else:
            return None


class EventRecognizer(BaseRecognizer):
    """Multi-class multi-label detection

    """
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        hop_length_seconds : float
            Analysis frame hop length in seconds.
            Default value 0.02

        params : dict
            Processing parameters

        class_labels : list of str
            Class labels in a list

        """

        super(EventRecognizer, self).__init__(*args, **kwargs)

        self.hop_length_seconds = kwargs.get('hop_length_seconds', 0.02)
        self.method = 'event'
        self.logger = kwargs.get('logger', logging.getLogger(__name__))

    def process(self, frame_probabilities):

        if isinstance(frame_probabilities, tuple) and len(frame_probabilities) == 2:
            return self.process_ratio(
                frame_probabilities_positive=frame_probabilities[0],
                frame_probabilities_negative=frame_probabilities[1],
            )
        else:
            return self.process_matrix(
                frame_probabilities=frame_probabilities
            )

    def process_matrix(self, frame_probabilities):
        """Multi-class multi-label detection.

        Parameters
        ----------
        frame_probabilities : numpy.ndarray
            Frame probabilities

        Returns
        -------
        results : str
            class label

        """

        # Accumulation
        if self.params.get_path('frame_accumulation.enable'):
            for event_id, event_label in enumerate(self.class_labels):
                frame_probabilities[event_id, :] = self.collapse_probabilities_windowed(
                    probabilities=frame_probabilities[event_id, :],
                    window_length=self.params.get_path('frame_accumulation.window_length_frames'),
                    operator=self.params.get_path('frame_accumulation.type'),
                )

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            # Binarization
            if self.params.get_path('frame_binarization.enable'):
                if self.params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities[event_id, :] > self.params.get_path('frame_binarization.threshold', 0.5)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=self.params.get_path('frame_binarization.type')
                    )
                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise AssertionError(message)

            # Processing
            if self.params.get_path('event_activity_processing.enable'):
                event_activity = self.process_activity(
                    activity_array=event_activity,
                    window_length=self.params.get_path('event_activity_processing.window_length_frames')
                )

            # Convert active frames into segments and translate frame indices into time stamps
            event_segments = self.find_contiguous_regions(event_activity) * self.hop_length_seconds

            # Store events
            for event in event_segments:
                results.append(
                    MetaDataItem(
                        {
                            'event_onset': event[0],
                            'event_offset': event[1],
                            'event_label': event_label
                        }
                    )
                )

        results = MetaDataContainer(results)

        if self.params.get_path('event_post_processing.enable'):
            # Event list post-processing
            results = results.process_events(
                minimum_event_length=self.params.get_path('event_post_processing.minimum_event_length_seconds'),
                minimum_event_gap=self.params.get_path('event_post_processing.minimum_event_gap_seconds')
            )

        return results

    def process_ratio(self, frame_probabilities_positive, frame_probabilities_negative):
        """Multi-class multi-label detection using likelihood ratio.

        Parameters
        ----------
        frame_probabilities_positive : numpy.ndarray
            Positive model frame probabilities

        frame_probabilities_negative : numpy.ndarray
            Negative model frame probabilities

        Returns
        -------
        results : str
            class label

        """

        results = MetaDataContainer()
        for event_id, event_label in enumerate(self.class_labels):
            # Accumulate
            event_frame_probabilities_positive = frame_probabilities_positive[event_id, :]
            event_frame_probabilities_negative = frame_probabilities_negative[event_id, :]

            positive_valid = not numpy.all(numpy.isnan(event_frame_probabilities_positive))
            negative_valid = not numpy.all(numpy.isnan(event_frame_probabilities_negative))

            if self.params.get_path('frame_accumulation.enable'):
                # Positive
                if positive_valid:
                    event_frame_probabilities_positive = self.collapse_probabilities_windowed(
                        probabilities=event_frame_probabilities_positive,
                        window_length=self.params.get_path('frame_accumulation.window_length_frames'),
                        operator=self.params.get_path('frame_accumulation.type')
                    )

                # Negative
                if negative_valid:
                    event_frame_probabilities_negative = self.collapse_probabilities_windowed(
                        probabilities=event_frame_probabilities_negative,
                        window_length=self.params.get_path('frame_accumulation.window_length_frames'),
                        operator=self.params.get_path('frame_accumulation.type')
                    )

            # Likelihood ratio
            if positive_valid and negative_valid:
                event_frame_probabilities = event_frame_probabilities_positive - event_frame_probabilities_negative

            elif not positive_valid and negative_valid:
                event_frame_probabilities = -event_frame_probabilities_negative

            elif positive_valid and not negative_valid:
                event_frame_probabilities = event_frame_probabilities_positive

            # Binarization
            if self.params.get_path('frame_binarization.enable'):
                if self.params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = event_frame_probabilities > self.params.get_path('frame_binarization.threshold', 0.0)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=self.params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise AssertionError(message)

            # Get events
            event_segments = self.find_contiguous_regions(event_activity) * self.hop_length_seconds

            # Add events
            for event in event_segments:
                results.append(
                    MetaDataItem(
                        {
                            'event_onset': event[0],
                            'event_offset': event[1],
                            'event_label': event_label
                        }
                    )
                )

        # Event list post-processing
        if self.params.get_path('event_post_processing.enable'):
            results = results.process_events(
                minimum_event_length=self.params.get_path('event_post_processing.minimum_event_length_seconds'),
                minimum_event_gap=self.params.get_path('event_post_processing.minimum_event_gap_seconds')
            )

        return results

