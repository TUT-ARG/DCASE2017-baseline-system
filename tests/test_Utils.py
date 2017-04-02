""" Unit tests for Utils """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.utils import get_parameter_hash
from nose.tools import *
import tempfile
import os


def test_get_parameter_hash():
    data = {
        'field1': {
            '1': [1, 2, 3],
            '2': 1234,
        },
        'field2': {
            'sub_field1': 1234
        }
    }
    data_hash_target = '064e6628408f570b9b5904f0af5228f5'

    nose.tools.eq_(get_parameter_hash(data), data_hash_target)

    data = {
        'field2': {
            'sub_field1': 1234
        },
        'field1': {
            '2': 1234,
            '1': [1, 2, 3],
        }
    }
    nose.tools.eq_(get_parameter_hash(data), data_hash_target)
