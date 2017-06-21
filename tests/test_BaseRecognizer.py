""" Unit tests for BaseRecognizer """

import nose.tools
import sys
sys.path.append('..')
import json
import os
import numpy
from dcase_framework.features import FeatureContainer, FeatureExtractor
from dcase_framework.metadata import MetaDataItem
from dcase_framework.recognizers import BaseRecognizer
import tempfile


def test_find_contiguous_regions():
    br = BaseRecognizer(
        class_labels=['event1', 'event2'],
        hop_length_seconds=0.02,
    )
    activity_array = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    res = br.find_contiguous_regions(activity_array=activity_array).tolist()
    nose.tools.assert_list_equal(res, [[4, 8], [12, 14], [16, 20]])


def test_slide_and_accumulate():
    br = BaseRecognizer(
        class_labels=['event1', 'event2'],
        hop_length_seconds=0.02,
    )
    probabilities = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Test sum
    res = br.collapse_probabilities_windowed(
        probabilities=probabilities,
        window_length=2,
        operator='sliding_sum'
    ).tolist()

    nose.tools.assert_list_equal(res, [3, 5, 7, 9, 11, 13, 15, 17, 9, 10])

    # Test mean
    res = br.collapse_probabilities_windowed(
        probabilities=probabilities,
        window_length=2,
        operator='sliding_mean'
    ).tolist()

    nose.tools.assert_list_equal(res, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0])

    # Test median
    res = br.collapse_probabilities_windowed(
        probabilities=probabilities,
        window_length=3,
        operator='sliding_median'
    ).tolist()

    nose.tools.assert_list_equal(res, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0])

    nose.tools.assert_raises(AssertionError, br.collapse_probabilities_windowed, probabilities, 2, "test")


def test_activity_processing():

    br = BaseRecognizer(
        class_labels=['event1', 'event2'],
        hop_length_seconds=0.02,
    )
    activity = numpy.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    res = br.process_activity(
        activity_array=activity,
        window_length=3,
        operator="median_filtering"
    ).tolist()
    nose.tools.assert_list_equal(res, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    nose.tools.assert_raises(AssertionError, br.process_activity, activity, 3, "test")

def test_accumulate_probabilities():

    br = BaseRecognizer(
        class_labels=['scene1', 'scene2'],
    )

    probabilities = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    # Test sum
    res = br.collapse_probabilities(
        probabilities=probabilities,
        operator='sum'
    )
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 55)
    nose.tools.eq_(res[1], 65)

    # Test mean
    res = br.collapse_probabilities(
        probabilities=probabilities,
        operator='mean'
    )
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 5.5)
    nose.tools.eq_(res[1], 6.5)

    # Test prod
    res = br.collapse_probabilities(
        probabilities=probabilities,
        operator='prod'
    )
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 3628800)
    nose.tools.eq_(res[1], 39916800)

    nose.tools.assert_raises(AssertionError, br.collapse_probabilities, probabilities, "test")

