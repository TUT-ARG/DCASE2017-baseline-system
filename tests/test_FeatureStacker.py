""" Unit tests for FeatureStacker """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.features import FeatureStacker, FeatureRepository, FeatureNormalizer, FeatureContainer, FeatureExtractor
from dcase_framework.parameters import ParameterContainer
import os
import tempfile


def test_process():
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

    # Test #1
    test_recipe = 'mfcc=0-5'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)

    feature_repository = FeatureRepository(filename_list={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_container = feature_stacker.process(feature_repository=feature_repository)

    nose.tools.assert_list_equal(sorted(list(feature_container.keys())), ['feat', 'meta', 'stat'])

    nose.tools.eq_(feature_container.channels, 1)
    nose.tools.eq_(feature_container.frames, 501)
    nose.tools.eq_(feature_container.vector_length, 6)

    nose.tools.eq_(feature_container.meta['audio_file'], 'material/test.wav')

    # Stat
    nose.tools.eq_(feature_container.stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_container.stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_container.feat[0].shape[0], 501)
    nose.tools.eq_(feature_container.feat[0].shape[1], 6)

    nose.tools.eq_(feature_container.shape[0], 501)
    nose.tools.eq_(feature_container.shape[1], 6)

    # Test #2
    test_recipe = 'mfcc=1,2,3,4'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)

    feature_repository = FeatureRepository(filename_list={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_container = feature_stacker.process(feature_repository=feature_repository)

    nose.tools.assert_list_equal(sorted(list(feature_container.keys())), ['feat', 'meta', 'stat'])

    nose.tools.eq_(feature_container.channels, 1)
    nose.tools.eq_(feature_container.frames, 501)
    nose.tools.eq_(feature_container.vector_length, 4)

    nose.tools.eq_(feature_container.meta['audio_file'], 'material/test.wav')

    # Stat
    nose.tools.eq_(feature_container.stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_container.stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_container.feat[0].shape[0], 501)
    nose.tools.eq_(feature_container.feat[0].shape[1], 4)

    nose.tools.eq_(feature_container.shape[0], 501)
    nose.tools.eq_(feature_container.shape[1], 4)


    # Test #1
    test_recipe = 'mfcc'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)

    feature_repository = FeatureRepository(filename_list={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_container = feature_stacker.process(feature_repository=feature_repository)

    nose.tools.assert_list_equal(sorted(list(feature_container.keys())), ['feat', 'meta', 'stat'])

    nose.tools.eq_(feature_container.channels, 1)
    nose.tools.eq_(feature_container.frames, 501)
    nose.tools.eq_(feature_container.vector_length, 10)

    nose.tools.eq_(feature_container.meta['audio_file'], 'material/test.wav')

    # Stat
    nose.tools.eq_(feature_container.stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_container.stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_container.feat[0].shape[0], 501)
    nose.tools.eq_(feature_container.feat[0].shape[1], 10)

    nose.tools.eq_(feature_container.shape[0], 501)
    nose.tools.eq_(feature_container.shape[1], 10)


def test_normalizer():
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
    test_recipe = 'mfcc=0-5'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_normalizer = FeatureNormalizer().accumulate(feature_container=feature_container).finalize()

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_normalizer = feature_stacker.normalizer(normalizer_list={'mfcc': feature_normalizer})

    nose.tools.eq_(feature_normalizer['N'][0][0], 501)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[1], 6)

    nose.tools.eq_(feature_normalizer['std'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['std'][0].shape[1], 6)

    # Test 2
    test_recipe = 'mfcc=1,2,3,4'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_normalizer = FeatureNormalizer().accumulate(feature_container=feature_container).finalize()

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_normalizer = feature_stacker.normalizer(normalizer_list={'mfcc': feature_normalizer})

    nose.tools.eq_(feature_normalizer['N'][0][0], 501)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[1], 4)

    nose.tools.eq_(feature_normalizer['std'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['std'][0].shape[1], 4)

    # Test 3
    test_recipe = 'mfcc'
    test_recipe_parsed = ParameterContainer()._parse_recipe(recipe=test_recipe)
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_normalizer = FeatureNormalizer().accumulate(feature_container=feature_container).finalize()

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_normalizer = feature_stacker.normalizer(normalizer_list={'mfcc': feature_normalizer})

    nose.tools.eq_(feature_normalizer['N'][0][0], 501)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['mean'][0].shape[1], 10)

    nose.tools.eq_(feature_normalizer['std'][0].shape[0], 1)
    nose.tools.eq_(feature_normalizer['std'][0].shape[1], 10)

