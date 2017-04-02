""" Unit tests for MetadataContainer """

import nose.tools
import sys
import numpy
sys.path.append('..')
from dcase_framework.metadata import MetaDataContainer
import tempfile

content = [
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'event_onset': 1.0,
            'event_offset': 10.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'mouse clicking',
            'event_onset': 3.0,
            'event_offset': 5.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'printer',
            'event_onset': 7.0,
            'event_offset': 9.0,
        },
        {
            'file': 'audio_002.wav',
            'scene_label': 'meeting',
            'event_label': 'speech',
            'event_onset': 1.0,
            'event_offset': 9.0,
        },
        {
            'file': 'audio_002.wav',
            'scene_label': 'meeting',
            'event_label': 'printer',
            'event_onset': 5.0,
            'event_offset': 7.0,
        },
    ]

content2 = [
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'event_onset': 1.0,
            'event_offset': 1.2,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'event_onset': 1.5,
            'event_offset': 3.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'event_onset': 4.0,
            'event_offset': 6.0,
        },
        {
            'file': 'audio_001.wav',
            'scene_label': 'office',
            'event_label': 'speech',
            'event_onset': 7.0,
            'event_offset': 8.0,
        },
    ]


def test_formats():
    delimiters = [',', ';', '\t']
    for delimiter in delimiters:
        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp')
        try:
            tmp.write('0.5' + delimiter + '0.7\n')
            tmp.write('2.5' + delimiter + '2.7\n')
            tmp.seek(0)
            nose.tools.assert_dict_equal(MetaDataContainer().load(filename=tmp.name)[0], {'event_offset': 0.7,
                                                                                     'event_onset': 0.5})
        finally:
            tmp.close()

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt',  dir='/tmp')
        try:
            tmp.write('0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.write('2.5' + delimiter + '2.7' + delimiter + 'event\n')
            tmp.seek(0)
            nose.tools.assert_dict_equal(MetaDataContainer().load(filename=tmp.name)[0],
                                         {'event_offset': 0.7,
                                          'event_onset': 0.5,
                                          'event_label': 'event'})
        finally:
            tmp.close()

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp')
        try:
            tmp.write('file' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.write('file' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event\n')
            tmp.seek(0)
            nose.tools.assert_dict_equal(MetaDataContainer().load(filename=tmp.name)[0],
                                         {'file': 'file',
                                          'scene_label': 'scene',
                                          'event_offset': 0.7,
                                          'event_onset': 0.5,
                                          'event_label': 'event'})
        finally:
            tmp.close()

        tmp = tempfile.NamedTemporaryFile('r+', suffix='.txt', dir='/tmp')
        try:
            tmp.write('file' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event' + delimiter + 'm' + delimiter + 'a1\n')
            tmp.write('file' + delimiter + 'scene' + delimiter + '0.5' + delimiter + '0.7' + delimiter + 'event' + delimiter + 'm' + delimiter + 'a2\n')
            tmp.seek(0)
            nose.tools.assert_dict_equal(MetaDataContainer().load(filename=tmp.name)[0],
                                         {'file': 'file',
                                          'scene_label': 'scene',
                                          'event_offset': 0.7,
                                          'event_onset': 0.5,
                                          'event_label': 'event',
                                          'identifier': 'a1',
                                          'source_label': 'm',
                                          })
        finally:
            tmp.close()


def test_content():
    meta = MetaDataContainer(content)
    nose.tools.eq_(len(meta), 5)

    # Test content
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.0)
    nose.tools.eq_(meta[0].event_offset, 10.0)

    nose.tools.eq_(meta[4].file, 'audio_002.wav')
    nose.tools.eq_(meta[4].scene_label, 'meeting')
    nose.tools.eq_(meta[4].event_label, 'printer')
    nose.tools.eq_(meta[4].event_onset, 5.0)
    nose.tools.eq_(meta[4].event_offset, 7.0)


def test_filter():
    # Test filter by file
    meta = MetaDataContainer(content).filter(filename='audio_002.wav')

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_002.wav')
    nose.tools.eq_(meta[0].scene_label, 'meeting')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.0)
    nose.tools.eq_(meta[0].event_offset, 9.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'printer')
    nose.tools.eq_(meta[1].event_onset, 5.0)
    nose.tools.eq_(meta[1].event_offset, 7.0)

    # Test filter by scene_label
    meta = MetaDataContainer(content).filter(scene_label='office')

    nose.tools.eq_(len(meta), 3)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.0)
    nose.tools.eq_(meta[0].event_offset, 10.0)

    nose.tools.eq_(meta[1].file, 'audio_001.wav')
    nose.tools.eq_(meta[1].scene_label, 'office')
    nose.tools.eq_(meta[1].event_label, 'mouse clicking')
    nose.tools.eq_(meta[1].event_onset, 3.0)
    nose.tools.eq_(meta[1].event_offset, 5.0)

    # Test filter by event_label
    meta = MetaDataContainer(content).filter(event_label='speech')

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.0)
    nose.tools.eq_(meta[0].event_offset, 10.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'speech')
    nose.tools.eq_(meta[1].event_onset, 1.0)
    nose.tools.eq_(meta[1].event_offset, 9.0)


def test_filter_time_segment():
    meta = MetaDataContainer(content).filter_time_segment(onset=5.0)

    nose.tools.eq_(len(meta), 2)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'printer')
    nose.tools.eq_(meta[0].event_onset, 7.0)
    nose.tools.eq_(meta[0].event_offset, 9.0)

    nose.tools.eq_(meta[1].file, 'audio_002.wav')
    nose.tools.eq_(meta[1].scene_label, 'meeting')
    nose.tools.eq_(meta[1].event_label, 'printer')
    nose.tools.eq_(meta[1].event_onset, 5.0)
    nose.tools.eq_(meta[1].event_offset, 7.0)

    meta = MetaDataContainer(content).filter_time_segment(onset=5.0, offset=7.0)

    nose.tools.eq_(len(meta), 1)
    nose.tools.eq_(meta[0].file, 'audio_002.wav')
    nose.tools.eq_(meta[0].scene_label, 'meeting')
    nose.tools.eq_(meta[0].event_label, 'printer')
    nose.tools.eq_(meta[0].event_onset, 5.0)
    nose.tools.eq_(meta[0].event_offset, 7.0)


def test_process_events():
    meta = MetaDataContainer(content2).process_events(minimum_event_gap=0.5, minimum_event_length=1.0)

    nose.tools.eq_(len(meta), 3)

    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.5)
    nose.tools.eq_(meta[0].event_offset, 3.0)

    nose.tools.eq_(meta[1].file, 'audio_001.wav')
    nose.tools.eq_(meta[1].scene_label, 'office')
    nose.tools.eq_(meta[1].event_label, 'speech')
    nose.tools.eq_(meta[1].event_onset, 4.0)
    nose.tools.eq_(meta[1].event_offset, 6.0)

    nose.tools.eq_(meta[2].file, 'audio_001.wav')
    nose.tools.eq_(meta[2].scene_label, 'office')
    nose.tools.eq_(meta[2].event_label, 'speech')
    nose.tools.eq_(meta[2].event_onset, 7.0)
    nose.tools.eq_(meta[2].event_offset, 8.0)

    meta = MetaDataContainer(content2).process_events(minimum_event_gap=1.0, minimum_event_length=1.0)

    nose.tools.eq_(len(meta), 1)
    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 1.5)
    nose.tools.eq_(meta[0].event_offset, 8.0)


def test_add_time_offset():
    meta = MetaDataContainer(content2).add_time_offset(offset=2.0)

    nose.tools.eq_(len(meta), 4)

    nose.tools.eq_(meta[0].file, 'audio_001.wav')
    nose.tools.eq_(meta[0].scene_label, 'office')
    nose.tools.eq_(meta[0].event_label, 'speech')
    nose.tools.eq_(meta[0].event_onset, 3.0)
    nose.tools.eq_(meta[0].event_offset, 3.2)

    nose.tools.eq_(meta[3].file, 'audio_001.wav')
    nose.tools.eq_(meta[3].scene_label, 'office')
    nose.tools.eq_(meta[3].event_label, 'speech')
    nose.tools.eq_(meta[3].event_onset, 9.0)
    nose.tools.eq_(meta[3].event_offset, 10.0)


def test_addition():
    meta = MetaDataContainer(content)
    meta2 = MetaDataContainer(content2)

    meta += meta2

    nose.tools.eq_(len(meta), 9)
    nose.tools.eq_(meta[8].file, 'audio_001.wav')
    nose.tools.eq_(meta[8].scene_label, 'office')
    nose.tools.eq_(meta[8].event_label, 'speech')
    nose.tools.eq_(meta[8].event_onset, 7.0)
    nose.tools.eq_(meta[8].event_offset, 8.0)


def test_is_number():
    meta = MetaDataContainer()

    nose.tools.eq_(meta._is_number('0.1'), True)
    nose.tools.eq_(meta._is_number('-2.1'), True)
    nose.tools.eq_(meta._is_number('123'), True)
    nose.tools.eq_(meta._is_number('-123'), True)
    nose.tools.eq_(meta._is_number('0'), True)

    nose.tools.eq_(meta._is_number('A'), False)
    nose.tools.eq_(meta._is_number('A123'), False)
    nose.tools.eq_(meta._is_number('A 123'), False)
    nose.tools.eq_(meta._is_number('AabbCc'), False)
    nose.tools.eq_(meta._is_number('A.2'), False)


def test_file_list():
    files = MetaDataContainer(content).file_list

    nose.tools.eq_(len(files), 2)
    nose.tools.eq_(files[0], 'audio_001.wav')
    nose.tools.eq_(files[1], 'audio_002.wav')


def test_event_count():
    nose.tools.eq_(MetaDataContainer(content).event_count, len(content))


def test_scene_label_count():
    nose.tools.eq_(MetaDataContainer(content).scene_label_count, 2)


def test_event_label_count():
    nose.tools.eq_(MetaDataContainer(content).event_label_count, 3)


def test_unique_event_labels():
    events = MetaDataContainer(content).unique_event_labels
    nose.tools.eq_(len(events), 3)
    nose.tools.eq_(events[0], 'mouse clicking')
    nose.tools.eq_(events[1], 'printer')
    nose.tools.eq_(events[2], 'speech')


def test_unique_scene_labels():
    scenes = MetaDataContainer(content).unique_scene_labels
    nose.tools.eq_(len(scenes), 2)
    nose.tools.eq_(scenes[0], 'meeting')
    nose.tools.eq_(scenes[1], 'office')


def test_max_event_offset():
    nose.tools.eq_(MetaDataContainer(content).max_event_offset, 10)



#embed()