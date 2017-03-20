""" Unit tests for EventRoll """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.metadata import MetaDataContainer, EventRoll
from IPython import embed
import tempfile


def test_construction():
    minimal_event_list = [
        {'event_label': 'A', 'event_onset': 0, 'event_offset': 1, },
        {'event_label': 'A', 'event_onset': 5, 'event_offset': 15, },
        {'event_label': 'B', 'event_onset': 1, 'event_offset': 2, },
        {'event_label': 'B', 'event_onset': 4, 'event_offset': 5, },
        {'event_label': 'C', 'event_onset': 7, 'event_offset': 12, }
    ]
    meta = MetaDataContainer(minimal_event_list)

    target_event_roll = numpy.array([
        [1., 0., 0.],  # 0
        [0., 1., 0.],  # 1
        [0., 0., 0.],  # 2
        [0., 0., 0.],  # 3
        [0., 1., 0.],  # 4
        [1., 0., 0.],  # 5
        [1., 0., 0.],  # 6
        [1., 0., 1.],  # 7
        [1., 0., 1.],  # 8
        [1., 0., 1.],  # 9
        [1., 0., 1.],  # 10
        [1., 0., 1.],  # 11
        [1., 0., 0.],  # 12
        [1., 0., 0.],  # 13
        [1., 0., 0.],  # 14
    ])

    # Test #1
    event_roll = EventRoll(metadata_container=meta, label_list=['A', 'B', 'C'], time_resolution=1.0).roll

    numpy.testing.assert_array_equal(target_event_roll, event_roll)
    nose.tools.assert_equal(event_roll.shape[0], target_event_roll.shape[0])
    nose.tools.assert_equal(event_roll.shape[1], target_event_roll.shape[1])


def test_pad():
    minimal_event_list = [
        {'event_label': 'A', 'event_onset': 0, 'event_offset': 1, },
        {'event_label': 'A', 'event_onset': 5, 'event_offset': 15, },
        {'event_label': 'B', 'event_onset': 1, 'event_offset': 2, },
        {'event_label': 'B', 'event_onset': 4, 'event_offset': 5, },
        {'event_label': 'C', 'event_onset': 7, 'event_offset': 12, }
    ]
    meta = MetaDataContainer(minimal_event_list)

    target_event_roll = numpy.array([
        [1., 0., 0.],  # 0
        [0., 1., 0.],  # 1
        [0., 0., 0.],  # 2
        [0., 0., 0.],  # 3
        [0., 1., 0.],  # 4
        [1., 0., 0.],  # 5
        [1., 0., 0.],  # 6
        [1., 0., 1.],  # 7
        [1., 0., 1.],  # 8
        [1., 0., 1.],  # 9
        [1., 0., 1.],  # 10
        [1., 0., 1.],  # 11
        [1., 0., 0.],  # 12
        [1., 0., 0.],  # 13
        [1., 0., 0.],  # 14
        [0., 0., 0.],  # 15
    ])

    # Test #1
    event_roll = EventRoll(metadata_container=meta, label_list=['A', 'B', 'C'], time_resolution=1.0).pad(length=16)

    numpy.testing.assert_array_equal(target_event_roll, event_roll)
    nose.tools.assert_equal(event_roll.shape[0], target_event_roll.shape[0])
    nose.tools.assert_equal(event_roll.shape[1], target_event_roll.shape[1])
