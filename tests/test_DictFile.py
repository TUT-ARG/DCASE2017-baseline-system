""" Unit tests for DictFile """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.files import DictFile
from nose.tools import *
import os
import tempfile
import pickle
import msgpack

from IPython import embed


def test_load():
    # YAML
    tmp = tempfile.NamedTemporaryFile('r+', suffix='.yaml',  dir='/tmp')
    try:
        tmp.write('section:\n')
        tmp.write('  field1: 1\n')
        tmp.write('  field2: 2\n')
        tmp.seek(0)

        m = DictFile().load(filename=tmp.name)

        nose.tools.assert_dict_equal(m, {'section': {'field1': 1, 'field2': 2}})
    finally:
        tmp.close()

    # Json
    tmp = tempfile.NamedTemporaryFile('r+', suffix='.json', dir='/tmp')
    try:
        tmp.write('{"section":{"field1":1,"field2":2}}\n')
        tmp.seek(0)

        m = DictFile().load(filename=tmp.name)

        nose.tools.assert_dict_equal(m, {'section': {'field1': 1, 'field2': 2}})
    finally:
        tmp.close()

    # pickle
    tmp = tempfile.NamedTemporaryFile('rb+', suffix='.pickle', dir='/tmp')
    try:
        data = {
            'section': {
                'field1': 1,
                'field2': 2
            }
        }
        pickle.dump(data, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.seek(0)

        m = DictFile().load(filename=tmp.name)

        nose.tools.assert_dict_equal(m, {'section': {'field1': 1, 'field2': 2}})
    finally:
        tmp.close()

    # msgpack
    tmp = tempfile.NamedTemporaryFile('rb+', suffix='.msgpack', dir='/tmp')
    try:
        data = {
            'section': {
                'field1': 1,
                'field2': 2
            }
        }
        msgpack.dump(data, tmp)
        tmp.seek(0)

        m = DictFile().load(filename=tmp.name)

        nose.tools.assert_dict_equal(m, {b'section': {b'field1': 1, b'field2': 2}})
    finally:
        tmp.close()

    # Txt
    tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp')
    try:
        tmp.write('line1\n')
        tmp.write('line2\n')
        tmp.write('line3\n')
        tmp.seek(0)

        m = DictFile().load(filename=tmp.name)

        nose.tools.assert_dict_equal(m, {0: 'line1\n', 1: 'line2\n', 2: 'line3\n'})
    finally:
        tmp.close()


def test_save():
    # Empty content
    DictFile({}).save(filename=os.path.join('material', 'saved.yaml'))

    # Content
    data = {
        'section1': {
            'field1': 1,
            'field2': [1, 2, 3, 4]
        },
        'section1': {
            'field1': {
                'field1': [1, 2, 3, 4]
            },
            'field2': [1, 2, 3, 4]
        }
    }
    DictFile(data).save(filename=os.path.join('material', 'saved.yaml'))
    d = DictFile().load(filename=os.path.join('material', 'saved.yaml'))

    nose.tools.assert_dict_equal(d, data)


def test_empty():
    # Test #1
    d = DictFile({})
    nose.tools.eq_(d.empty(), True)

    # Test #2
    d = DictFile({'sec':1})
    nose.tools.eq_(d.empty(), False)


@raises(IOError)
def test_load_not_found():
    DictFile().load(filename=os.path.join('material', 'wrong.cpickle'))


@raises(IOError)
def test_load_wrong_type():
    DictFile().load(filename=os.path.join('material', 'wrong.wav'))


@raises(IOError)
def test_load_wrong_type2():
    DictFile().load(filename=os.path.join('material', 'wrong.abc'))
