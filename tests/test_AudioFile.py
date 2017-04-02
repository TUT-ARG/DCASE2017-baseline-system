""" Unit tests for DictFile """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.files import AudioFile
from nose.tools import *
import tempfile
import os


def test_load():
    # Mono
    audio_data_mono, fs = AudioFile(filename=os.path.join('material', 'test.wav'),  mono=True).load()

    nose.tools.eq_(fs, 44100)
    nose.tools.eq_(len(audio_data_mono.shape), 1)
    nose.tools.eq_(audio_data_mono.shape[0], 441001)

    # Stereo
    audio_data_stereo, fs = AudioFile(filename=os.path.join('material', 'test.wav'), mono=False).load()

    nose.tools.eq_(fs, 44100)
    nose.tools.eq_(audio_data_stereo.shape[0], 2)
    nose.tools.eq_(audio_data_stereo.shape[1], 441001)

    # Resampling
    audio_data_mono, fs = AudioFile(filename=os.path.join('material', 'test.wav'), fs=16000, mono=True).load()
    nose.tools.eq_(fs, 16000)
    nose.tools.eq_(len(audio_data_mono.shape), 1)
    nose.tools.eq_(audio_data_mono.shape[0], 160001)
