""" Unit tests for SceneClassifier """

import nose.tools
import sys
sys.path.append('..')
import json
import os
import numpy
from dcase_framework.features import FeatureContainer, FeatureExtractor
from dcase_framework.metadata import MetaDataItem
from dcase_framework.learners import SceneClassifier
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
        'file1.wav': MetaDataItem(
            {
                'file': 'file1.wav',
                'scene_label': 'scene1',
            }
        ),
        'file2.wav': MetaDataItem(
            {
                'file': 'file2.wav',
                'scene_label': 'scene2',
            }
        ),
    }

    sc = SceneClassifier(
        class_labels=['scene1', 'scene2'],
        disable_progress_bar=True,
    )
    target_matrix = sc._get_target_matrix_dict(data=data, annotations=annotations)

    # Test shape
    nose.tools.eq_(target_matrix['file1.wav'].shape, (501, 2))
    nose.tools.eq_(target_matrix['file2.wav'].shape, (501, 2))

    # Test content
    nose.tools.eq_(numpy.any(target_matrix['file1.wav'][:, 0] == 1), True)
    nose.tools.eq_(numpy.any(target_matrix['file1.wav'][:, 1] == 1), False)

    nose.tools.eq_(numpy.any(target_matrix['file2.wav'][:, 0] == 1), False)
    nose.tools.eq_(numpy.any(target_matrix['file2.wav'][:, 1] == 1), True)


def test_accumulate_probabilities():
    sc = SceneClassifier(class_labels=['scene1', 'scene2'])

    probabilities = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    # Test sum
    res = sc._accumulate_probabilities(probabilities=probabilities, accumulation_type='sum')
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 55)
    nose.tools.eq_(res[1], 65)

    # Test mean
    res = sc._accumulate_probabilities(probabilities=probabilities, accumulation_type='mean')
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 5.5)
    nose.tools.eq_(res[1], 6.5)

    # Test prod
    res = sc._accumulate_probabilities(probabilities=probabilities, accumulation_type='prod')
    nose.tools.eq_(res.shape, (2,))
    nose.tools.eq_(res[0], 3628800)
    nose.tools.eq_(res[1], 39916800)

    nose.tools.assert_raises(AssertionError, sc._accumulate_probabilities, probabilities, "test")


def test_generate_validation():

    annotations = {
        'file1.wav': MetaDataItem(
            {
                'file': 'file1.wav',
                'scene_label': 'scene1',
                'location_identifier': 'a',
            }
        ),
        'file2.wav': MetaDataItem(
            {
                'file': 'file2.wav',
                'scene_label': 'scene1',
                'location_identifier': 'b',
            }
        ),
        'file3.wav': MetaDataItem(
            {
                'file': 'file3.wav',
                'scene_label': 'scene1',
                'location_identifier': 'c',
            }
        ),
        'file4.wav': MetaDataItem(
            {
                'file': 'file4.wav',
                'scene_label': 'scene2',
                'location_identifier': 'd',
            }
        ),
        'file5.wav': MetaDataItem(
            {
                'file': 'file5.wav',
                'scene_label': 'scene2',
                'location_identifier': 'e',
            }
        ),
        'file6.wav': MetaDataItem(
            {
                'file': 'file6.wav',
                'scene_label': 'scene2',
                'location_identifier': 'f',
            }
        ),
    }
    sc = SceneClassifier(class_labels=['scene1', 'scene2'])

    validation_set = sc._generate_validation(
        annotations=annotations,
        validation_type='generated_scene_balanced',
        valid_percentage=0.50, seed=0
    )
    nose.tools.eq_(len(validation_set), 4)

    nose.tools.assert_raises(AssertionError, sc._generate_validation, annotations, 'test', 0.5)

