""" Unit tests for LearnerContainer """

import nose.tools
import sys
sys.path.append('..')
import json
import os
from dcase_framework.learners import LearnerContainer
import tempfile
from IPython import embed


def test_init():

    class_labels = ['class1', 'class2']

    model = {
            'model': 1234,
        }
    params = {
            'param1': 1,
            'param2': 2,
            'param3': 3,
            'param4': 4,
            'seed': 1
        }
    lc = LearnerContainer(
        method='test',
        class_labels=class_labels,
        params=params,
        model=model
    )
    # Test method
    nose.tools.eq_(lc.method, 'test')

    # Test class labels
    nose.tools.assert_list_equal(lc.class_labels, class_labels)

    # Test model
    nose.tools.assert_dict_equal(lc.model, model)

    # Test params
    nose.tools.assert_dict_equal(lc.params, params)
    nose.tools.assert_dict_equal(lc.learner_params, params)

    # Seed from parameters
    nose.tools.eq_(lc.seed, params['seed'])

    # Seed as argument
    params = {
            'param1': 1,
            'param2': 2,
            'param3': 3,
            'param4': 4,
        }
    lc = LearnerContainer(
        method='test',
        class_labels=class_labels,
        params=params,
        model=model,
        seed=1234
    )
    nose.tools.eq_(lc.seed, 1234)

    # Feature masker
    nose.tools.eq_(lc.feature_masker, None)

    # Feature stacker
    nose.tools.eq_(lc.feature_stacker, None)

    # Feature aggregator
    nose.tools.eq_(lc.feature_aggregator, None)

