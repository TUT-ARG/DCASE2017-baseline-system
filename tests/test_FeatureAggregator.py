""" Unit tests for FeatureAggregator """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.features import FeatureAggregator, FeatureStacker, FeatureContainer, FeatureExtractor
import tempfile
import os


def test():
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
    feature_aggregator = FeatureAggregator(
        recipe=['mean'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})
    feature_matrix = feature_stacker.process(feature_data=feature_repository)
    feature_matrix = feature_aggregator.process(feature_data=feature_matrix)

    nose.tools.eq_(feature_matrix.shape[0], 501)
    nose.tools.eq_(feature_matrix.shape[1], 10)

    # Test #2
    feature_aggregator = FeatureAggregator(
        recipe=['mean', 'std'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})
    feature_matrix = feature_stacker.process(feature_data=feature_repository)
    feature_matrix = feature_aggregator.process(feature_data=feature_matrix)

    nose.tools.eq_(feature_matrix.shape[0], 501)
    nose.tools.eq_(feature_matrix.shape[1], 2*10)

    # Test #3
    feature_aggregator = FeatureAggregator(
        recipe=['mean', 'std', 'kurtosis', 'skew'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})
    feature_matrix = feature_stacker.process(feature_data=feature_repository)
    feature_matrix = feature_aggregator.process(feature_data=feature_matrix)

    nose.tools.eq_(feature_matrix.shape[0], 501)
    nose.tools.eq_(feature_matrix.shape[1], 4*10)

    # Test #4
    feature_aggregator = FeatureAggregator(
        recipe=['cov'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})
    feature_matrix = feature_stacker.process(feature_data=feature_repository)
    feature_matrix = feature_aggregator.process(feature_data=feature_matrix)

    nose.tools.eq_(feature_matrix.shape[0], 501)
    nose.tools.eq_(feature_matrix.shape[1], 10*10)

    # Test #5
    feature_aggregator = FeatureAggregator(
        recipe=['flatten'],
        win_length_frames=10,
        hop_length_frames=1,
    )

    feature_stacker = FeatureStacker(recipe=[{'method': 'mfcc'}])
    feature_repository = FeatureContainer().load(filename_dict={'mfcc': os.path.join('material', 'test.mfcc.cpickle')})
    feature_matrix = feature_stacker.process(feature_data=feature_repository)
    feature_matrix = feature_aggregator.process(feature_data=feature_matrix)

    nose.tools.eq_(feature_matrix.shape[0], 501)
    nose.tools.eq_(feature_matrix.shape[1], 10*10)
