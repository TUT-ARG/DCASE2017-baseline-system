""" Unit tests for FeatureMasker """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.features import FeatureMasker, FeatureRepository, FeatureExtractor
from dcase_framework.metadata import MetaDataContainer
import tempfile
import os
from IPython import embed


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
    feature_masker = FeatureMasker(hop_length_seconds=1)
    mask_events = MetaDataContainer([
        {
            'event_onset': 10.0,
            'event_offset': 50.0,
        }
    ])
    feature_repository = FeatureRepository().load(filename_list={'mfcc1': os.path.join('material', 'test.mfcc.cpickle'),
                                                                 'mfcc2': os.path.join('material', 'test.mfcc.cpickle')})
    original_length = feature_repository['mfcc1'].shape[0]
    feature_masker.process(feature_repository=feature_repository, mask_events=mask_events)

    nose.tools.eq_(feature_repository['mfcc1'].shape[0], original_length-40)

    # Test #2
    feature_masker = FeatureMasker(hop_length_seconds=1)
    mask_events = MetaDataContainer([
        {
            'event_onset': 10.0,
            'event_offset': 50.0,
        },
        {
           'event_onset': 120.0,
           'event_offset': 150.0,
        },
    ])
    feature_repository = FeatureRepository().load(filename_list={'mfcc1': os.path.join('material', 'test.mfcc.cpickle'),
                                                                 'mfcc2': os.path.join('material',
                                                                                       'test.mfcc.cpickle')})
    original_length = feature_repository['mfcc1'].shape[0]
    feature_masker.process(feature_repository=feature_repository, mask_events=mask_events)

    nose.tools.eq_(feature_repository['mfcc1'].shape[0], original_length - 40 - 30)
