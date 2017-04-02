""" Unit tests for DictFile """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.files import ListFile
from nose.tools import *
import tempfile
import os


def test_load():
    # Txt
    tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', prefix='prefix_', dir='/tmp')
    try:
        tmp.write('line1\n')
        tmp.write('line2\n')
        tmp.write('line3\n')
        tmp.seek(0)

        m = ListFile().load(filename=tmp.name)

        nose.tools.assert_list_equal(m, ['line1', 'line2', 'line3'])
    finally:
        tmp.close()


def test_save():
    ListFile(['line1', 'line2', 'line3']).save(filename=os.path.join('material', 'saved.txt'))
    d = ListFile().load(filename=os.path.join('material', 'saved.txt'))
    nose.tools.assert_list_equal(d, ['line1', 'line2', 'line3'])

    f = open(os.path.join('material', 'saved.txt'), 'r')
    x = f.readlines()
    nose.tools.assert_list_equal(x, ['line1\n', 'line2\n', 'line3\n'])


def test_empty():
    # Test #1
    d = ListFile([])
    nose.tools.eq_(d.empty(), True)

    # Test #2
    d = ListFile(['line1','line2'])
    nose.tools.eq_(d.empty(), False)


@raises(IOError)
def test_load_not_found2():
    ListFile().load(filename=os.path.join('material', 'wrong.txt'))


@raises(IOError)
def test_load_wrong_type():
    ListFile().load(filename=os.path.join('material', 'wrong.cpickle'))


@raises(IOError)
def test_load_wrong_type2():
    ListFile().load(filename=os.path.join('material', 'wrong.abc'))
