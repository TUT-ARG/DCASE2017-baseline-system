""" Unit tests for FeatureContainer """

import nose.tools
import sys
import numpy
sys.path.append('..')
import os
from dcase_framework.features import FeatureContainer
from nose.tools import *
import tempfile
from IPython import embed


def test_load():
    # Test #1
    feature_container = FeatureContainer(filename=os.path.join('material', 'test.mfcc.cpickle'))

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_container.keys())), ['feat', 'meta', 'stat'])

    nose.tools.eq_(feature_container.channels, 1)
    nose.tools.eq_(feature_container.frames, 501)
    nose.tools.eq_(feature_container.vector_length, 10)

    nose.tools.eq_(feature_container.meta['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_container.meta['parameters']['n_mels'], 40)
    nose.tools.eq_(feature_container.meta['parameters']['n_mfcc'], 10)

    # Stat
    nose.tools.eq_(feature_container.stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_container.stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_container.feat[0].shape[0], 501)
    nose.tools.eq_(feature_container.feat[0].shape[1], 10)

    nose.tools.eq_(feature_container.shape[0], 501)
    nose.tools.eq_(feature_container.shape[1], 10)

    # Test #2
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_container.keys())), ['feat', 'meta', 'stat'])

    nose.tools.eq_(feature_container.meta['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_container.meta['parameters']['n_mels'], 40)
    nose.tools.eq_(feature_container.meta['parameters']['n_mfcc'], 10)

    # Stat
    nose.tools.eq_(feature_container.stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_container.stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_container.feat[0].shape[0], 501)
    nose.tools.eq_(feature_container.feat[0].shape[1], 10)

    nose.tools.eq_(feature_container.shape[0], 501)
    nose.tools.eq_(feature_container.shape[1], 10)

    # Test #3
    feature_repository = FeatureContainer().load(filename_list={'mfcc1': os.path.join('material', 'test.mfcc.cpickle'),
                                                                'mfcc2': os.path.join('material', 'test.mfcc.cpickle')})

    nose.tools.assert_list_equal(sorted(list(feature_repository.keys())), ['mfcc1', 'mfcc2'])


def test_empty():
    nose.tools.eq_(FeatureContainer().shape, None)
    nose.tools.eq_(FeatureContainer().channels, None)
    nose.tools.eq_(FeatureContainer().frames, None)
    nose.tools.eq_(FeatureContainer().vector_length, None)
    nose.tools.eq_(FeatureContainer().feat, None)
    nose.tools.eq_(FeatureContainer().stat, None)
    nose.tools.eq_(FeatureContainer().meta, None)


@raises(IOError)
def test_load_not_found():
    FeatureContainer().load(filename=os.path.join('material', 'wrong.cpickle'))


@raises(IOError)
def test_load_wrong_type():
    FeatureContainer().load(filename=os.path.join('material', 'wrong.yaml'))


@raises(IOError)
def test_load_wrong_type2():
    FeatureContainer().load(filename=os.path.join('material', 'wrong.txt'))


def test_save():
    feature_container = FeatureContainer().load(filename=os.path.join('material', 'test.mfcc.cpickle'))
    feature_container.save(filename=os.path.join('material', 'saved.mfcc.cpickle'))

test_load()