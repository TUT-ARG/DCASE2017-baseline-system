""" Unit tests for FeatureNormalizer """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.features import FeatureNormalizer, FeatureContainer, FeatureExtractor
import os


def test_accumulate_finalize():
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

    # Test 1
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_normalizer = FeatureNormalizer().accumulate(feature_container=feature_container).finalize()

    nose.tools.eq_(feature_normalizer['N'][0], 501)

    numpy.testing.assert_array_equal(feature_normalizer['mean'][0][0],
                                     numpy.mean(feature_container.feat[0], axis=0))
    numpy.testing.assert_array_equal(feature_normalizer['S1'][0],
                                     numpy.sum(feature_container.feat[0], axis=0))
    numpy.testing.assert_array_equal(feature_normalizer['S2'][0],
                                     numpy.sum(feature_container.feat[0]**2, axis=0))

    # Test 2
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_normalizer = FeatureNormalizer()
    feature_normalizer.accumulate(feature_container=feature_container)
    feature_normalizer.accumulate(feature_container=feature_container)
    feature_normalizer.finalize()

    nose.tools.eq_(feature_normalizer['N'][0], 501*2)

    numpy.testing.assert_array_equal(feature_normalizer['mean'][0][0],
                                     numpy.mean(feature_container.feat[0], axis=0))
    numpy.testing.assert_array_equal(feature_normalizer['S1'][0],
                                     numpy.sum(feature_container.feat[0], axis=0)*2)
    numpy.testing.assert_array_equal(feature_normalizer['S2'][0],
                                     numpy.sum(feature_container.feat[0] ** 2, axis=0)*2)


def test_with_statement():
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

    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    with FeatureNormalizer() as feature_normalizer:
        feature_normalizer.accumulate(feature_container)

    nose.tools.eq_(feature_normalizer['N'][0], 501)

    numpy.testing.assert_array_equal(feature_normalizer['mean'][0][0],
                                     numpy.mean(feature_container.feat[0], axis=0))
    numpy.testing.assert_array_equal(feature_normalizer['S1'][0],
                                     numpy.sum(feature_container.feat[0], axis=0))
    numpy.testing.assert_array_equal(feature_normalizer['S2'][0],
                                     numpy.sum(feature_container.feat[0] ** 2, axis=0))

    test_accumulate_finalize()