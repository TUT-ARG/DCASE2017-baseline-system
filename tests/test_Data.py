""" Unit tests for Data """

import nose.tools
import sys
import numpy
sys.path.append('..')
import os
from dcase_framework.data import DataBuffer, DataSequencer, DataProcessor, ProcessingChain
from dcase_framework.features import FeatureStacker, FeatureRepository, FeatureNormalizer, FeatureContainer, FeatureExtractor
from dcase_framework.parameters import ParameterContainer
from nose.tools import *
import numpy


def test_data_buffer():
    data = {
        'key1': {
            'data': numpy.ones((10, 10)),
            'meta': 'label1'
        },
        'key2': {
            'data': numpy.ones((10, 10))*2,
            'meta': 'label2'
        },
        'key3': {
            'data': numpy.ones((10, 10))*3,
            'meta': 'label3'
        },
        'key4': {
            'data': numpy.ones((10, 10))*4,
            'meta': 'label4'
        }
    }
    keys = sorted(list(data.keys()))

    # Same size buffer
    db = DataBuffer(size=4)
    for key in keys:
        db.set(key=key, data=data[key]['data'], meta=data[key]['meta'])

    for key in keys:
        nose.tools.eq_(db.key_exists(key=key), True)

    for key in keys:
        item_data, item_meta = db.get(key=key)

        numpy.testing.assert_array_equal(item_data, data[key]['data'])
        nose.tools.eq_(item_meta, data[key]['meta'])

    # Smaller size buffer
    db = DataBuffer(size=2)

    for key in keys:
        db.set(key=key, data=data[key]['data'], meta=data[key]['meta'])

    for i in range(0, 2):
        nose.tools.eq_(db.key_exists(key=keys[i]), False)

    for i in range(2, 4):
        nose.tools.eq_(db.key_exists(key=keys[i]), True)

    for i in range(2, 4):
        key = keys[i]
        item_data, item_meta = db.get(key=key)

        numpy.testing.assert_array_equal(item_data, data[key]['data'])
        nose.tools.eq_(item_meta, data[key]['meta'])


def test_data_sequencer():
    # Perfect fitting
    data = [numpy.arange(0, 10), numpy.arange(0, 10), numpy.arange(0, 10)]
    data = numpy.vstack(data).T

    ds = DataSequencer(frames=2, hop=2, padding=False)
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 5)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[0, 0, 0], [1, 1, 1]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[2, 2, 2], [3, 3, 3]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[4, 4, 4], [5, 5, 5]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[6, 6, 6], [7, 7, 7]]))
    numpy.testing.assert_array_equal(seq[4], numpy.array([[8, 8, 8], [9, 9, 9]]))

    # Non-perfect fitting without padding
    data = [numpy.arange(0, 11), numpy.arange(0, 11), numpy.arange(0, 11)]
    data = numpy.vstack(data).T
    ds = DataSequencer(frames=2, hop=2, padding=False)
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 5)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[0, 0, 0], [1, 1, 1]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[2, 2, 2], [3, 3, 3]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[4, 4, 4], [5, 5, 5]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[6, 6, 6], [7, 7, 7]]))
    numpy.testing.assert_array_equal(seq[4], numpy.array([[8, 8, 8], [9, 9, 9]]))

    # Non-perfect fitting with padding
    data = [numpy.arange(0, 11), numpy.arange(0, 11), numpy.arange(0, 11)]
    data = numpy.vstack(data).T
    ds = DataSequencer(frames=2, hop=2, padding=True)
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 6)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[0, 0, 0], [1, 1, 1]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[2, 2, 2], [3, 3, 3]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[4, 4, 4], [5, 5, 5]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[6, 6, 6], [7, 7, 7]]))
    numpy.testing.assert_array_equal(seq[4], numpy.array([[8, 8, 8], [9, 9, 9]]))
    numpy.testing.assert_array_equal(seq[5], numpy.array([[10, 10, 10], [10, 10, 10]]))

    # Non-perfect fitting without padding, and roll shifting
    data = [numpy.arange(0, 11), numpy.arange(0, 11), numpy.arange(0, 11)]
    data = numpy.vstack(data).T
    ds = DataSequencer(frames=2, hop=2, padding=False, shift_step=1, shift_border='roll')
    ds.increase_shifting()
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 5)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[1, 1, 1], [2, 2, 2]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[3, 3, 3], [4, 4, 4]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[5, 5, 5], [6, 6, 6]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[7, 7, 7], [8, 8, 8]]))

    ds.increase_shifting()
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 5)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[2, 2, 2], [3, 3, 3]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[4, 4, 4], [5, 5, 5]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[6, 6, 6], [7, 7, 7]]))

    # Non-perfect fitting without padding, and push shifting
    data = [numpy.arange(0, 11), numpy.arange(0, 11), numpy.arange(0, 11)]
    data = numpy.vstack(data).T
    ds = DataSequencer(frames=2, hop=2, padding=False, shift_step=1, shift_border='shift')
    ds.increase_shifting()
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 5)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[1, 1, 1], [2, 2, 2]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[3, 3, 3], [4, 4, 4]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[5, 5, 5], [6, 6, 6]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[7, 7, 7], [8, 8, 8]]))
    numpy.testing.assert_array_equal(seq[4], numpy.array([[9, 9, 9], [10, 10, 10]]))

    ds.increase_shifting()
    seq = ds.process(data)
    nose.tools.eq_(len(seq), 4)

    numpy.testing.assert_array_equal(seq[0], numpy.array([[2, 2, 2], [3, 3, 3]]))
    numpy.testing.assert_array_equal(seq[1], numpy.array([[4, 4, 4], [5, 5, 5]]))
    numpy.testing.assert_array_equal(seq[2], numpy.array([[6, 6, 6], [7, 7, 7]]))
    numpy.testing.assert_array_equal(seq[3], numpy.array([[8, 8, 8], [9, 9, 9]]))


def test_processing_chain():
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

    feature_repository = FeatureRepository(
        filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')}
    )

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_container = feature_stacker.process(feature_data=feature_repository)

    feature_chain = ProcessingChain()
    feature_chain.append(feature_stacker)

    feature_container_chain = feature_chain.process(data=feature_repository)

    numpy.testing.assert_array_equal(feature_container.feat, feature_container_chain.feat)


def test_data_processor():
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

    feature_repository = FeatureRepository(
        filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')}
    )

    feature_stacker = FeatureStacker(recipe=test_recipe_parsed)
    feature_container = feature_stacker.process(feature_data=feature_repository)

    ds = DataSequencer(frames=10, hop=10, padding=False)
    target_data = ds.process(data=feature_container.feat[0])

    dp = DataProcessor(
        feature_processing_chain=ProcessingChain([feature_stacker]),
        data_processing_chain=ProcessingChain([ds])
    )
    processed_data, feature_matrix_size = dp.process(
        feature_data=feature_repository
    )

    numpy.testing.assert_array_equal(target_data, processed_data[:, 0, :, :])
