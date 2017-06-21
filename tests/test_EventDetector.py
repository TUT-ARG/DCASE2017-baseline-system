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


def test_generate_validation():

    annotations = {
        'file1.wav': MetaDataContainer([
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event1',
                    'identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event2',
                    'identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 2.0,
                    'event_offset': 3.0,
                    'event_label': 'event2',
                    'identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 4.0,
                    'event_offset': 5.0,
                    'event_label': 'event1',
                    'identifier': 'a',
                },
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event1',
                    'identifier': 'a',
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
                    'identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 1.0,
                    'event_offset': 2.0,
                    'event_label': 'event1',
                    'identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 2.0,
                    'event_offset': 3.0,
                    'event_label': 'event2',
                    'identifier': 'b',
                },
                {
                    'file': 'file2.wav',
                    'scene_label': 'scene1',
                    'event_onset': 3.0,
                    'event_offset': 4.0,
                    'event_label': 'event2',
                    'identifier': 'b',
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
    nose.tools.eq_(len(validation_set), 1)

    # Test generated_event_file_balanced
    annotations = {
        'file1.wav': MetaDataContainer([
                {
                    'file': 'file1.wav',
                    'scene_label': 'scene1',
                    'event_onset': 0.0,
                    'event_offset': 1.0,
                    'event_label': 'event1',
                    'identifier': 'a',
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
                    'identifier': 'b',
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

