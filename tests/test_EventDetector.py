""" Unit tests for SceneClassifier """

import nose.tools
import sys
sys.path.append('..')
import json
import os
import numpy
from dcase_framework.features import FeatureContainer, FeatureExtractor
from dcase_framework.metadata import MetaDataContainer, MetaDataItem
from dcase_framework.learners import EventDetector
import tempfile
from IPython import embed


def test_get_target_matrix_dict():
    FeatureExtractor(store=True, overwrite=True).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name='mfcc',
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        },
        storage_paths={
            'mfcc': os.path.join('material', 'test.mfcc.cpickle')
        }
    )

    feature_container = FeatureContainer(filename=os.path.join('material', 'test.mfcc.cpickle'))

    data = {
        'file1.wav': feature_container,
        'file2.wav': feature_container,
    }

    annotations = {
        'file1.wav': MetaDataContainer([
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event1',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event2',
                }
            ]
        ),
        'file2.wav': MetaDataContainer([
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event2',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event1',
                }
            ]
        ),
    }

    ed = EventDetector(
        class_labels=['event1', 'event2'],
        disable_progress_bar=True,
        params={
            'hop_length_seconds': 0.02,
        }
    )
    target_matrix = ed._get_target_matrix_dict(data=data, annotations=annotations)

    # Test shape
    nose.tools.eq_(target_matrix['file1.wav'].shape, (501, 2))
    nose.tools.eq_(target_matrix['file2.wav'].shape, (501, 2))

    # Test content
    nose.tools.eq_(numpy.sum(target_matrix['file1.wav'][:, 0] == 1), 50)
    nose.tools.eq_(numpy.sum(target_matrix['file1.wav'][:, 1] == 1), 50)

    nose.tools.eq_(numpy.sum(target_matrix['file2.wav'][:, 0] == 1), 50)
    nose.tools.eq_(numpy.sum(target_matrix['file2.wav'][:, 1] == 1), 50)


def test_contiguous_regions():
    ed = EventDetector(
        class_labels=['event1', 'event2'],
        disable_progress_bar=True,
        params={
            'hop_length_seconds': 0.02,
        }
    )
    activity_array = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    res = ed._contiguous_regions(activity_array=activity_array).tolist()
    nose.tools.assert_list_equal(res, [[4, 8], [12, 14], [16, 20]])


def test_slide_and_accumulate():

    ed = EventDetector(
        class_labels=['event1', 'event2'],
        disable_progress_bar=True,
        params={
            'hop_length_seconds': 0.02,
        }
    )
    probabilities = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Test sum
    res = ed._slide_and_accumulate(input_probabilities=probabilities, window_length=2, accumulation_type='sliding_sum').tolist()
    nose.tools.assert_list_equal(res, [3, 5, 7, 9, 11, 13, 15, 17, 9, 10])

    # Test mean
    res = ed._slide_and_accumulate(input_probabilities=probabilities, window_length=2, accumulation_type='sliding_mean').tolist()
    nose.tools.assert_list_equal(res, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0])

    # Test median
    res = ed._slide_and_accumulate(input_probabilities=probabilities, window_length=3, accumulation_type='sliding_median').tolist()
    nose.tools.assert_list_equal(res, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0])

    nose.tools.assert_raises(AssertionError, ed._slide_and_accumulate, probabilities, 2, "test")


def test_activity_processing():

    ed = EventDetector(
        class_labels=['event1', 'event2'],
        disable_progress_bar=True,
        params={
            'hop_length_seconds': 0.02,
        }
    )
    activity = numpy.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    res = ed._activity_processing(activity_vector=activity, window_size=3, processing_type="median_filtering").tolist()
    nose.tools.assert_list_equal(res, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    nose.tools.assert_raises(AssertionError, ed._activity_processing, activity, 3, "test")


def test_generate_validation():


    annotations = {
        'file1.wav': MetaDataContainer([
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event1',
                    'location_identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event2',
                    'location_identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 2.0,
                    'event_offset': 3.0,
                    'event_label': 'event2',
                    'location_identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 4.0,
                    'event_offset': 5.0,
                    'event_label': 'event1',
                    'location_identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event1',
                    'location_identifier': 'a',
                }
            ]
        ),
        'file2.wav': MetaDataContainer([
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event2',
                    'location_identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event1',
                    'location_identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 2.0,
                    'event_offset': 3.0,
                    'event_label': 'event2',
                    'location_identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 3.0,
                    'event_offset': 4.0,
                    'event_label': 'event2',
                    'location_identifier': 'b',
                }
            ]
        )
    }

    ed = EventDetector(
        class_labels=['event1', 'event2'],
        disable_progress_bar=True,
        params={
            'hop_length_seconds': 0.02,
        }
    )

    # Test generated_scene_location_event_balanced
    validation_set = ed._generate_validation(
             annotations=annotations,
             validation_type='generated_scene_location_event_balanced',
             valid_percentage=0.50, seed=0
    )
    if sys.version_info[0] < 3:
        nose.tools.assert_list_equal(validation_set, ['file1.wav'])
    else:
        nose.tools.assert_list_equal(validation_set, ['file2.wav'])

    # Test generated_event_file_balanced
    annotations = {
        'file1.wav': MetaDataContainer([
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event1',
                    'location_identifier': 'a',
                },
            ]
        ),
        'file2.wav': MetaDataContainer([
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event2',
                    'location_identifier': 'b',
                },
            ]
        )
    }

    validation_set = ed._generate_validation(
             annotations=annotations,
             validation_type='generated_event_file_balanced',
             valid_percentage=0.50, seed=0
    )
    nose.tools.assert_list_equal(validation_set, ['file1.wav', 'file2.wav'])

    nose.tools.assert_raises(AssertionError, ed._generate_validation, annotations, 'test', 0.5)

