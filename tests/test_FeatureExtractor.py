""" Unit tests for FeatureExtractor """

import nose.tools
import sys
import numpy
sys.path.append('..')
from nose.tools import *
from dcase_framework.features import FeatureExtractor
import os
import tempfile
from IPython import embed


def test_extract():
    # MFCC
    extractor_name = 'mfcc'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mfcc': {
                'n_mfcc': 12
            }
        }
    )

    nose.tools.eq_(len(feature_repository), 1)
    nose.tools.assert_list_equal(sorted(list(feature_repository.keys())), [extractor_name])

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].keys())), ['feat', 'meta', 'stat'])
    nose.tools.eq_(feature_repository[extractor_name]['meta']['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['n_mfcc'], 12)

    # Stat
    nose.tools.eq_(feature_repository[extractor_name].stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].stat[0].keys())), ['N','S1', 'S2','mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[1], 12)

    nose.tools.eq_(feature_repository[extractor_name].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].shape[1], 12)

    # MFCC - delta
    extractor_name = 'mfcc_delta'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mfcc': {
                'n_mfcc': 12
            }
        }
    )

    nose.tools.eq_(len(feature_repository), 1)
    nose.tools.assert_list_equal(list(feature_repository.keys()), [extractor_name])

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].keys())), ['feat', 'meta', 'stat'])
    nose.tools.eq_(feature_repository[extractor_name]['meta']['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['dependency_method'], 'mfcc')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['dependency_parameters']['n_mfcc'], 12)

    # Stat
    nose.tools.eq_(feature_repository[extractor_name].stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].stat[0].keys())),['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[1], 12)

    nose.tools.eq_(feature_repository[extractor_name].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].shape[1], 12)

    # MFCC - acceleration
    extractor_name = 'mfcc_acceleration'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mfcc': {
                'n_mfcc': 12
            }
        }
    )

    nose.tools.eq_(len(feature_repository), 1)
    nose.tools.assert_list_equal(list(feature_repository.keys()), [extractor_name])

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].keys())), ['feat', 'meta', 'stat'])
    nose.tools.eq_(feature_repository[extractor_name]['meta']['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['dependency_method'], 'mfcc')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['dependency_parameters']['n_mfcc'], 12)

    # Stat
    nose.tools.eq_(feature_repository[extractor_name].stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[1], 12)

    nose.tools.eq_(feature_repository[extractor_name].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].shape[1], 12)

    # MEL
    extractor_name = 'mel'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mel': {
                'n_mels': 10
            }
        }
    )

    nose.tools.eq_(len(feature_repository), 1)
    nose.tools.assert_list_equal(list(feature_repository.keys()), [extractor_name])

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].keys())), ['feat', 'meta', 'stat'])
    nose.tools.eq_(feature_repository[extractor_name]['meta']['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['n_mels'], 10)

    # Stat
    nose.tools.eq_(feature_repository[extractor_name].stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[1], 10)

    nose.tools.eq_(feature_repository[extractor_name].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].shape[1], 10)

    # MFCC
    extractor_name = 'mfcc'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_params={
            'mfcc': {
                'n_mfcc': 12
            }
        }
    )

    nose.tools.eq_(len(feature_repository), 1)
    nose.tools.assert_list_equal(list(feature_repository.keys()), [extractor_name])

    # Meta
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].keys())), ['feat', 'meta', 'stat'])
    nose.tools.eq_(feature_repository[extractor_name]['meta']['audio_file'], 'material/test.wav')
    nose.tools.eq_(feature_repository[extractor_name]['meta']['parameters']['n_mfcc'], 12)

    # Stat
    nose.tools.eq_(feature_repository[extractor_name].stat[0]['N'], 501)
    nose.tools.assert_list_equal(sorted(list(feature_repository[extractor_name].stat[0].keys())), ['N', 'S1', 'S2', 'mean', 'std'])

    # Feat
    # Shape
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].feat[0].shape[1], 12)

    nose.tools.eq_(feature_repository[extractor_name].shape[0], 501)
    nose.tools.eq_(feature_repository[extractor_name].shape[1], 12)

def test_save():
    extractor_name = 'mfcc'
    feature_repository = FeatureExtractor(store=True, overwrite=True).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        },
        storage_paths={
            'mfcc': os.path.join('material','test.mfcc.cpickle')
        }
    )

@raises(ValueError)
def test_wrong_extractor():
    extractor_name = 'mf'
    feature_repository = FeatureExtractor(store=False).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name=extractor_name,
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        }
    )
