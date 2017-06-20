""" Unit tests for FeatureRepository """

import nose.tools
import sys
import os
sys.path.append('..')
from dcase_framework.features import FeatureRepository
import tempfile


def test_load():
    feature_repository = FeatureRepository().load(
        filename_dict={'mfcc1': os.path.join('material', 'test.mfcc.cpickle'),
                       'mfcc2': os.path.join('material', 'test.mfcc.cpickle')}
    )

    nose.tools.assert_list_equal(sorted(list(feature_repository.keys())), ['mfcc1', 'mfcc2'])