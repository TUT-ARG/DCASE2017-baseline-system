#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta data
=========

Utility classes for meta data storage and handling.

Usage examples:

.. code-block:: python
    :linenos:

    # load meta data
    metadata_container = MetaDataContainer(filename='event_list.txt').load()

    # Print content
    print(metadata_container)

    # Log content
    metadata_container.log()

    # Filter based on event label
    print(metadata_container.filter(event_label='drawer'))

    # Filter based on scene label
    print(metadata_container.filter(scene_label='office'))

    # Filter time segment
    print(metadata_container.filter_time_segment(onset=10.0, offset=30.0))

    # Add time offset
    metadata_container.add_time_offset(100)

    # Process events
    print(metadata_container.process_events(minimum_event_length=0.5, minimum_event_gap=0.5))

    # Combine content
    metadata_container2 = MetaDataContainer(filename='event_list.txt').load()
    metadata_container += metadata_container2

    # Unique file list
    metadata_container.file_list

MetaDataItem
^^^^^^^^^^^^

Dict based class for storing meta data item (i.e. one row in meta data file).

.. autosummary::
    :toctree: generated/

    MetaDataItem
    MetaDataItem.id
    MetaDataItem.file
    MetaDataItem.scene_label
    MetaDataItem.event_label
    MetaDataItem.event_onset
    MetaDataItem.event_offset
    MetaDataItem.identifier
    MetaDataItem.source_label

MetaDataContainer
^^^^^^^^^^^^^^^^^

List of MetaDataItems for storing meta data file in one container.

Reads meta data from CSV-text files. Preferred delimiter is tab, however, other delimiters are supported automatically (they are sniffed automatically).

Supported input formats:

- [file(string)]
- [event_onset (float)][tab][event_offset (float)]
- [file(string)][scene_label(string)]
- [file(string)][scene_label(string)][identifier(string)]
- [event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
- [file(string)][event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
- [file(string)][event_onset (float)][tab][event_offset (float)][tab][event_label (string)][tab][identifier(string)]
- [file(string)[tab][scene_label][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
- [file(string)[tab][scene_label][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)][tab][source_label(string)]

.. autosummary::
    :toctree: generated/

    MetaDataContainer
    MetaDataContainer.log
    MetaDataContainer.show
    MetaDataContainer.get_string
    MetaDataContainer.update
    MetaDataContainer.filter
    MetaDataContainer.filter_time_segment
    MetaDataContainer.process_events
    MetaDataContainer.remove_field
    MetaDataContainer.slice_field
    MetaDataContainer.filter_time_segment
    MetaDataContainer.add_time_offset
    MetaDataContainer.file_list
    MetaDataContainer.event_count
    MetaDataContainer.scene_label_count
    MetaDataContainer.event_label_count
    MetaDataContainer.unique_scene_labels
    MetaDataContainer.unique_event_labels
    MetaDataContainer.max_offset
    MetaDataContainer.load
    MetaDataContainer.save
    MetaDataContainer.event_stat_counts
    MetaDataContainer.event_roll

EventRoll
^^^^^^^^^

Class to convert MetaDataContainer to binary matrix indicating event activity withing time segment defined by time_resolution.

.. autosummary::
    :toctree: generated/

    EventRoll
    EventRoll.roll
    EventRoll.pad
    EventRoll.plot


ProbabilityItem
^^^^^^^^^^^^^^^

Dict based class for storing meta data item along with probability.

.. autosummary::
    :toctree: generated/

    ProbabilityItem
    ProbabilityItem.id
    ProbabilityItem.file
    ProbabilityItem.label
    ProbabilityItem.timestamp
    ProbabilityItem.probability
    ProbabilityItem.get_list

ProbabilityContainer
^^^^^^^^^^^^^^^^^^^^

List of ProbabilityItem for storing meta data along with probabilities in one container.

.. autosummary::
    :toctree: generated/

    ProbabilityContainer
    ProbabilityContainer.log
    ProbabilityContainer.show
    ProbabilityContainer.update
    ProbabilityContainer.file_list
    ProbabilityContainer.unique_labels
    ProbabilityContainer.filter
    ProbabilityContainer.get_string
    ProbabilityContainer.load
    ProbabilityContainer.save

"""

from __future__ import print_function, absolute_import

from .files import ListFile
from .utils import posix_path, get_parameter_hash
import os
import numpy
import csv
import math
import logging
import copy
import re


class MetaMixin(object):

    @property
    def _delimiter(self):
        """Use csv.sniffer to guess delimeter for CSV file

        Returns
        -------

        """

        sniffer = csv.Sniffer()
        valid_delimiters = ['\t', ',', ';', ' ']
        delimiter = '\t'
        with open(self.filename, 'rt') as f1:
            try:
                example_content = f1.read(1024)
                dialect = sniffer.sniff(example_content)
                if hasattr(dialect, '_delimiter'):
                    if dialect._delimiter in valid_delimiters:
                        delimiter = dialect._delimiter
                elif hasattr(dialect, 'delimiter'):
                    if dialect.delimiter in valid_delimiters:
                        delimiter = dialect.delimiter
                else:
                    # Fall back to default
                    delimiter = '\t'
            except:
                # Fall back to default
                delimiter = '\t'
        return delimiter

    def update(self, data):
        """Replace content with given list

        Parameters
        ----------
        data : list
            New content

        Returns
        -------
        self

        """

        list.__init__(self, data)
        return self

    def log(self, level='info'):
        """Log container content

        Parameters
        ----------
        level : str
            Logging level, possible values [info, debug, warn, warning, error, critical]
            Default value "info"

        Returns
        -------
            Nothing

        """

        lines = str(self).split('\n')
        logger = logging.getLogger(__name__)
        for line in lines:
            if level.lower() == 'debug':
                logger.debug(line)
            elif level.lower() == 'info':
                logger.info(line)
            elif level.lower() == 'warn' or level.lower() == 'warning':
                logger.warn(line)
            elif level.lower() == 'error':
                logger.error(line)
            elif level.lower() == 'critical':
                logger.critical(line)

    def show(self, **kwargs):
        """Print container content

        Returns
        -------
            Nothing

        """

        print(self.get_string(**kwargs))


class MetaDataItem(dict):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
            dict

        """

        dict.__init__(self, *args)

        # Process fields
        if 'file' in self:
            # Keep file paths in unix format even under Windows
            self['file'] = posix_path(self['file'])

        if 'event_onset' in self:
            self['onset'] = self['event_onset']

        if 'event_offset' in self:
            self['offset'] = self['event_offset']

        if 'onset' in self:
            self['onset'] = float(self['onset'])
            self['event_onset'] = self['onset']

        if 'offset' in self:
            self['offset'] = float(self['offset'])
            self['event_offset'] = self['offset']

        if 'event_label' in self and self.event_label:
            self['event_label'] = self['event_label'].strip()
            if self['event_label'].lower() == 'none':
                self['event_label'] = None

        if 'scene_label' in self and self.scene_label:
            self['scene_label'] = self['scene_label'].strip()
            if self['scene_label'].lower() == 'none':
                self['scene_label'] = None

        if 'tags' in self and self.tags:
            if isinstance(self['tags'], str):
                self['tags'] = self['tags'].strip()
                if self['tags'].lower() == 'none':
                    self['tags'] = None

                if self['tags'] and '#' in self['tags']:
                    self['tags'] = [x.strip() for x in self['tags'].split('#')]
                elif self['tags'] and ',' in self['tags']:
                    self['tags'] = [x.strip() for x in self['tags'].split(',')]
                elif self['tags'] and ';' in self['tags']:
                    self['tags'] = [x.strip() for x in self['tags'].split(';')]
                elif self['tags'] and ':' in self['tags']:
                    self['tags'] = [x.strip() for x in self['tags'].split(':')]

            # Remove empty tags
            self['tags'] = list(filter(None, self['tags']))

            # Sort tags
            self['tags'].sort()

    def __str__(self):
        if len(self.file) > 30:
            file_string = '...'+self.file[-27:]
        else:
            file_string = self.file

        string_data = '  {0:<30s} |'.format(
            file_string if file_string is not None else '---'
        )

        if self.onset is not None:
            string_data += ' {:6.2f} |'.format(self.onset)
        else:
            string_data += ' {:>6s} |'.format('---')

        if self.offset is not None:
            string_data += ' {:6.2f} |'.format(self.offset)
        else:
            string_data += ' {:>6s} |'.format('---')

        string_data += ' {:<18s} |'.format(self.scene_label if self.scene_label is not None else '---')
        string_data += ' {:<20s} |'.format(self.event_label if self.event_label is not None else '---')
        string_data += ' {:<20s} |'.format(','.join(self.tags) if self.tags is not None else '---')
        string_data += ' {:<8s} |'.format(self.identifier if self.identifier is not None else '---')
        string_data += ' {:<8s} |'.format(self.source_label if self.source_label is not None else '---')

        return string_data

    def __setitem__(self, key, value):
        if key == 'event_onset':
            super(MetaDataItem, self).__setitem__('event_onset', value)
            return super(MetaDataItem, self).__setitem__('onset', value)
        elif key == 'event_offset':
            super(MetaDataItem, self).__setitem__('event_offset', value)
            return super(MetaDataItem, self).__setitem__('offset', value)
        else:
            return super(MetaDataItem, self).__setitem__(key, value)

    @property
    def id(self):
        """Unique item identifier

        ID is formed by taking MD5 hash of the item data.

        Returns
        -------
        id : str
            Unique item id
        """

        string = ''
        if self.file:
            string += self.file
        if self.scene_label:
            string += self.scene_label
        if self.event_label:
            string += self.event_label
        if self.tags:
            string += ','.join(self.tags)
        if self.onset:
            string += '{:8.4f}'.format(self.onset)
        if self.offset:
            string += '{:8.4f}'.format(self.offset)

        return get_parameter_hash(string)

    @staticmethod
    def get_header():
        string_data = '  {0:<30s} | {1:<6s} | {2:<6s} | {3:<18s} | {4:<20s} | {5:<20s} | {6:<8s} | {7:<8s} |\n'.format(
            'File',
            'Onset',
            'Offset',
            'Scene label',
            'Event label',
            'Tags',
            'Loc.ID',
            'Source'
        )

        string_data += '  {0:<30s} + {1:<6s} + {2:<6s} + {3:<18s} + {4:<20s} + {5:<20s} + {6:<8s} + {7:<8s} +\n'.format(
            '-'*30,
            '-'*6,
            '-'*6,
            '-' * 18,
            '-'*20,
            '-'*20,
            '-'*8,
            '-'*8
        )

        return string_data

    def get_list(self):
        """Return item values in a list with specified order.

        Returns
        -------
        list
        """
        fields = list(self.keys())

        # Select only valid fields
        valid_fields = ['event_label', 'file', 'offset', 'onset', 'scene_label', 'identifier', 'source_label', 'tags']
        fields = list(set(fields).intersection(valid_fields))
        fields.sort()

        if fields == ['file']:
            return [self.file]

        elif fields == ['event_label', 'file', 'offset', 'onset', 'scene_label']:
            return [self.file, self.scene_label, self.onset, self.offset, self.event_label]

        elif fields == ['offset', 'onset']:
            return [self.onset, self.offset]

        elif fields == ['event_label', 'offset', 'onset']:
            return [self.onset, self.offset, self.event_label]

        elif fields == ['file', 'scene_label']:
            return [self.file, self.scene_label]

        elif fields == ['file', 'identifier', 'scene_label']:
            return [self.file, self.scene_label, self.identifier]

        elif fields == ['event_label', 'file', 'offset', 'onset']:
            return [self.file, self.onset, self.offset, self.event_label]

        elif fields == ['event_label', 'file', 'offset', 'onset', 'identifier', 'scene_label']:
            return [self.file, self.scene_label, self.onset, self.offset, self.event_label, self.identifier]

        elif fields == ['event_label', 'file', 'offset', 'onset', 'scene_label', 'source_label']:
            return [self.file, self.scene_label, self.onset, self.offset, self.event_label, self.source_label]

        elif fields == ['event_label', 'file', 'offset', 'onset', 'identifier', 'scene_label', 'source_label']:
            return [self.file, self.scene_label, self.onset, self.offset, self.event_label,
                    self.source_label, self.identifier]

        elif fields == ['file', 'tags']:
            return [self.file, ";".join(self.tags)+";"]

        elif fields == ['file', 'scene_label', 'tags']:
            return [self.file, self.scene_label, ";".join(self.tags)+";"]

        elif fields == ['file', 'offset', 'onset', 'scene_label', 'tags']:
            return [self.file, self.scene_label, self.onset, self.offset, ";".join(self.tags)+";"]

        else:
            message = '{name}: Invalid meta data format [{format}]'.format(
                name=self.__class__.__name__,
                format=str(fields)
            )
            raise ValueError(message)

    @property
    def file(self):
        """Filename

        Returns
        -------
        str or None
            filename

        """

        if 'file' in self:
            return self['file']
        else:
            return None

    @file.setter
    def file(self, value):
        # Keep file paths in unix format even under Windows
        self['file'] = posix_path(value)

    @property
    def scene_label(self):
        """Scene label

        Returns
        -------
        str or None
            scene label

        """

        if 'scene_label' in self:
            return self['scene_label']
        else:
            return None

    @scene_label.setter
    def scene_label(self, value):
        self['scene_label'] = value

    @property
    def event_label(self):
        """Event label

        Returns
        -------
        str or None
            event label

        """

        if 'event_label' in self:
            return self['event_label']
        else:
            return None

    @event_label.setter
    def event_label(self, value):
        self['event_label'] = value

    @property
    def onset(self):
        """Onset

        Returns
        -------
        float or None
            onset

        """

        if 'onset' in self:
            return self['onset']
        else:
            return None

    @onset.setter
    def onset(self, value):
        self['onset'] = float(value)

    @property
    def offset(self):
        """Offset

        Returns
        -------
        float or None
            offset

        """

        if 'offset' in self:
            return self['offset']
        else:
            return None

    @offset.setter
    def offset(self, value):
        self['offset'] = float(value)

    @property
    def event_onset(self):
        """Event onset

        Returns
        -------
        float or None
            event onset

        """

        if 'onset' in self:
            return self['onset']
        else:
            return None

    @event_onset.setter
    def event_onset(self, value):
        self['onset'] = float(value)
        self['event_onset'] = self['onset']

    @property
    def event_offset(self):
        """Event offset

        Returns
        -------
        float or None
            event offset

        """

        if 'offset' in self:
            return self['offset']
        else:
            return None

    @event_offset.setter
    def event_offset(self, value):
        self['offset'] = float(value)
        self['event_offset'] = self['offset']

    @property
    def identifier(self):
        """Identifier

        Returns
        -------
        str or None
            location identifier

        """

        if 'identifier' in self:
            return self['identifier']
        else:
            return None

    @identifier.setter
    def identifier(self, value):
        self['identifier'] = value

    @property
    def source_label(self):
        """Source label

        Returns
        -------
        str or None
            source label

        """

        if 'source_label' in self:
            return self['source_label']
        else:
            return None

    @source_label.setter
    def source_label(self, value):
        self['source_label'] = value

    @property
    def tags(self):
        """Tags

        Returns
        -------
        list or None
            tags

        """

        if 'tags' in self:
            return self['tags']
        else:
            return None

    @tags.setter
    def tags(self, value):
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == 'none':
                value = None

            if value and '#' in value:
                value = [x.strip() for x in value.split('#')]
            elif value and ',' in value:
                value = [x.strip() for x in value.split(',')]
            elif value and ':' in value:
                value = [x.strip() for x in value.split(':')]
            elif value and ';' in value:
                value = [x.strip() for x in value.split(';')]

        self['tags'] = value

        # Remove empty tags
        self['tags'] = list(filter(None, self['tags']))

        # Sort tags
        self['tags'].sort()


class MetaDataContainer(ListFile, MetaMixin):
    valid_formats = ['csv', 'txt', 'ann']

    def __init__(self, *args, **kwargs):
        super(MetaDataContainer, self).__init__(*args, **kwargs)
        self.item_class = MetaDataItem

        # Convert all items in the list to MetaDataItems
        for item_id in range(0, len(self)):
            if not isinstance(self[item_id], self.item_class):
                self[item_id] = self.item_class(self[item_id])

    def __str__(self):
        return self.get_string()

    def __add__(self, other):
        return self.update(super(MetaDataContainer, self).__add__(other))

    def log(self, level='info'):
        """Log container content

        Parameters
        ----------
        level : str
            Logging level, possible values [info, debug, warn, warning, error, critical]
            Default value "info"

        Returns
        -------
            Nothing

        """

        lines = str(self).split('\n')
        logger = logging.getLogger(__name__)
        for line in lines:
            if level.lower() == 'debug':
                logger.debug(line)
            elif level.lower() == 'info':
                logger.info(line)
            elif level.lower() == 'warn' or level.lower() == 'warning':
                logger.warn(line)
            elif level.lower() == 'error':
                logger.error(line)
            elif level.lower() == 'critical':
                logger.critical(line)

    def show(self, show_stats=True):
        """Print container content

        Returns
        -------
            Nothing

        """

        print(self.get_string(show_stats=show_stats))

    def get_string(self, show_stats=True):
        """Get content in string format

        Parameters
        ----------
        show_stats : bool
            Include scene and event statistics
            Default value "True"

        Returns
        -------
        str
            Multi-line string

        """

        string_data = ''
        string_data += self.item_class().get_header()
        for i in self:
            string_data += str(self.item_class(i)) + '\n'
        stats = self._stats()

        if show_stats:
            if 'scenes' in stats and 'scene_label_list' in stats['scenes'] and stats['scenes']['scene_label_list']:
                string_data += '\n  === Scene statistics ===\n'
                string_data += '  {0:<40s} | {1:<5s} |\n'.format('Scene label', 'Count')
                string_data += '  {0:<40s} + {1:<5s} +\n'.format('-' * 40, '-' * 5)

                for scene_id, scene_label in enumerate(stats['scenes']['scene_label_list']):
                    string_data += '  {0:<40s} | {1:5d} |\n'.format(scene_label,
                                                                    int(stats['scenes']['count'][scene_id]))

            if 'events' in stats and 'event_label_list' in stats['events'] and stats['events']['event_label_list']:
                string_data += '\n  === Event statistics ===\n'
                string_data += '  {0:<40s} | {1:<5s} | {2:<10s} | {3:<10s} |\n'.format(
                    'Event label',
                    'Count',
                    'Total Len',
                    'Avg Len'
                )

                string_data += '  {0:<40s} + {1:<5s} + {2:10s} + {3:10s} +\n'.format(
                    '-'*40,
                    '-'*5,
                    '-'*10,
                    '-'*10
                )

                for event_id, event_label in enumerate(stats['events']['event_label_list']):
                    string_data += '  {0:<40s} | {1:5d} | {2:10.2f} | {3:10.2f} |\n'.format(
                        (event_label[:38] + '..') if len(event_label) > 38 else event_label,
                        int(stats['events']['count'][event_id]),
                        stats['events']['length'][event_id],
                        stats['events']['avg_length'][event_id]
                    )

            if 'tags' in stats and 'tag_list' in stats['tags'] and stats['tags']['tag_list']:
                string_data += '\n  === Tag statistics ===\n'
                string_data += '  {0:<40s} | {1:<5s} |\n'.format('Tag', 'Count')
                string_data += '  {0:<40s} + {1:<5s} +\n'.format('-' * 40, '-' * 5)

                for tag_id, tag in enumerate(stats['tags']['tag_list']):
                    string_data += '  {0:<40s} | {1:5d} |\n'.format(tag, int(stats['tags']['count'][tag_id]))

        return string_data

    def filter(self, filename=None, file_list=None, scene_label=None, event_label=None, tag=None):
        """Filter content

        Parameters
        ----------
        filename : str, optional
            Filename to be matched
            Default value "None"
        file_list : list, optional
            List of filenames to be matched
            Default value "None"
        scene_label : str, optional
            Scene label to be matched
            Default value "None"
        event_label : str, optional
            Event label to be matched
            Default value "None"
        tag : str, optional
            Tag to be matched
            Default value "None"

        Returns
        -------
        MetaDataContainer

        """

        data = []
        for item in self:
            matched = False
            if filename and item.file == filename:
                matched = True
            if file_list and item.file in file_list:
                matched = True
            if scene_label and item.scene_label == scene_label:
                matched = True
            if event_label and item.event_label == event_label:
                matched = True
            if tag and item.tags and tag in item.tags:
                matched = True

            if matched:
                data.append(copy.deepcopy(item))

        return MetaDataContainer(data)

    def filter_time_segment(self, onset=None, offset=None):
        """Filter time segment

        Parameters
        ----------
        onset : float > 0.0
            Segment start
            Default value "None"
        offset : float > 0.0
            Segment end
            Default value "None"

        Returns
        -------
        MetaDataContainer

        """

        data = []
        for item in self:
            matched = False
            if onset and not offset and item.onset >= onset:
                matched = True
            elif not onset and offset and item.offset <= offset:
                matched = True
            elif onset and offset and item.onset >= onset and item.offset <= offset:
                matched = True

            if matched:
                data.append(item)

        return MetaDataContainer(data)

    def process_events(self, minimum_event_length=None, minimum_event_gap=None):
        """Process event content

        Makes sure that minimum event length and minimum event gap conditions are met per event label class.

        Parameters
        ----------
        minimum_event_length : float > 0.0
            Minimum event length in seconds, shorten than given are filtered out from the output.
            (Default value=None)
        minimum_event_gap : float > 0.0
            Minimum allowed gap between events in seconds from same event label class.
            (Default value=None)

        Returns
        -------
        MetaDataContainer

        """

        processed_events = []

        for filename in self.file_list:

            for event_label in self.unique_event_labels:
                current_events_items = self.filter(filename=filename, event_label=event_label)

                # Sort events
                current_events_items = sorted(current_events_items, key=lambda k: k.event_onset)

                # 1. remove short events
                event_results_1 = []
                for event in current_events_items:
                    if minimum_event_length is not None:
                        if event.offset - event.onset >= minimum_event_length:
                            event_results_1.append(event)
                    else:
                        event_results_1.append(event)

                if len(event_results_1) and minimum_event_gap is not None:
                    # 2. remove small gaps between events
                    event_results_2 = []

                    # Load first event into event buffer
                    buffered_event_onset = event_results_1[0].onset
                    buffered_event_offset = event_results_1[0].offset
                    for i in range(1, len(event_results_1)):
                        if event_results_1[i].onset - buffered_event_offset > minimum_event_gap:
                            # The gap between current event and the buffered is bigger than minimum event gap,
                            # store event, and replace buffered event
                            current_event = copy.deepcopy(event_results_1[i])
                            current_event.onset = buffered_event_onset
                            current_event.offset = buffered_event_offset
                            event_results_2.append(current_event)

                            buffered_event_onset = event_results_1[i].onset
                            buffered_event_offset = event_results_1[i].offset
                        else:
                            # The gap between current event and the buffered is smaller than minimum event gap,
                            # extend the buffered event until the current offset
                            buffered_event_offset = event_results_1[i].offset

                    # Store last event from buffer
                    current_event = copy.copy(event_results_1[len(event_results_1) - 1])
                    current_event.onset = buffered_event_onset
                    current_event.offset = buffered_event_offset
                    event_results_2.append(current_event)

                    processed_events += event_results_2
                else:
                    processed_events += event_results_1

        return MetaDataContainer(processed_events)

    def add_time_offset(self, offset):
        """Add time offset to event onset and offset timestamps

        Parameters
        ----------
        offset : float > 0.0
            Offset to be added to the onset and offsets
        Returns
        -------

        """

        for item in self:
            if item.onset:
                item.onset += offset

            if item.offset:
                item.offset += offset

        return self

    def remove_field(self, field_name):
        """Remove field from meta items

        Parameters
        ----------
        field_name : str
            Field name
        Returns
        -------

        """

        for item in self:
            if field_name in item:
                del item[field_name]

        return self

    def slice_field(self, field_name):
        """Slice field values into list

        Parameters
        ----------
        field_name : str
            Field name
        Returns
        -------

        """

        data = []
        for item in self:
            if field_name in item:
                data.append(item[field_name])
            else:
                data.append(None)
        return data

    @property
    def file_list(self):
        """List of unique files in the container

        Returns
        -------
        list

        """

        files = {}
        for item in self:
            files[item.file] = item.file

        return sorted(files.values())

    @property
    def event_count(self):
        """Get number of events

        Returns
        -------
        event_count: integer > 0

        """

        return len(self)

    @property
    def scene_label_count(self):
        """Get number of unique scene labels

        Returns
        -------
        scene_label_count: float >= 0

        """

        return len(self.unique_scene_labels)

    @property
    def event_label_count(self):
        """Get number of unique event labels

        Returns
        -------
        event_label_count: float >= 0

        """

        return len(self.unique_event_labels)

    @property
    def unique_event_labels(self):
        """Get unique event labels

        Returns
        -------
        labels: list, shape=(n,)
            Unique labels in alphabetical order

        """

        labels = []
        for item in self:
            if 'event_label' in item and item['event_label'] not in labels:
                labels.append(item['event_label'])

        labels.sort()
        return labels

    @property
    def unique_scene_labels(self):
        """Get unique scene labels

        Returns
        -------
        labels: list, shape=(n,)
            Unique labels in alphabetical order

        """

        labels = []
        for item in self:
            if 'scene_label' in item and item['scene_label'] not in labels:
                labels.append(item['scene_label'])

        labels.sort()
        return labels

    @property
    def unique_tags(self):
        """Get unique tags

        Returns
        -------
        tags: list, shape=(n,)
            Unique tags in alphabetical order

        """

        tags = []
        for item in self:
            if 'tags' in item:
                for tag in item['tags']:
                    if tag not in tags:
                        tags.append(tag)

        tags.sort()
        return tags

    @property
    def max_offset(self):
        """Find the offset (end-time) of last event

        Returns
        -------
        max_offset: float > 0
            maximum offset

        """

        max_offset = 0
        for item in self:
            if 'offset' in item and item.offset > max_offset:
                max_offset = item.offset
        return max_offset

    def _stats(self, event_label_list=None, scene_label_list=None, tag_list=None):
        """Statistics of the container content

        Parameters
        ----------
        event_label_list : list of str
            List of event labels to be included in the statistics. If none given, all unique labels used
            Default value "None"
        scene_label_list : list of str
            List of scene labels to be included in the statistics. If none given, all unique labels used
            Default value "None"
        tag_list : list of str
            List of tags to be included in the statistics. If none given, all unique tags used
            Default value "None"

        Returns
        -------
        dict

        """

        if event_label_list is None:
            event_label_list = self.unique_event_labels

        if scene_label_list is None:
            scene_label_list = self.unique_scene_labels

        if tag_list is None:
            tag_list = self.unique_tags

        scene_counts = numpy.zeros(len(scene_label_list))

        for scene_id, scene_label in enumerate(scene_label_list):
            for item in self:
                if item.scene_label and item.scene_label == scene_label:
                    scene_counts[scene_id] += 1

        event_lengths = numpy.zeros(len(event_label_list))
        event_counts = numpy.zeros(len(event_label_list))

        for event_id, event_label in enumerate(event_label_list):
            for item in self:
                if item.onset is not None and item.offset is not None and item.event_label == event_label:
                    event_lengths[event_id] += item.offset - item.onset
                    event_counts[event_id] += 1

        tag_counts = numpy.zeros(len(tag_list))
        for tag_id, tag in enumerate(tag_list):
            for item in self:
                if item.tags and tag in item.tags:
                    tag_counts[tag_id] += 1

        return {
            'scenes': {
                'count': scene_counts,
                'scene_label_list': scene_label_list,
            },
            'events': {
                'length': event_lengths,
                'count': event_counts,
                'avg_length': event_lengths/event_counts,
                'event_label_list': event_label_list
            },
            'tags': {
                'count': tag_counts,
                'tag_list': tag_list,
            }
        }

    def load(self, filename=None):
        """Load event list from delimited text file (csv-formated)

        Preferred delimiter is tab, however, other delimiters are supported automatically (they are sniffed automatically).

        Supported input formats:
            - [file(string)]
            - [file(string)][scene_label(string)]
            - [file(string)][scene_label(string)][identifier(string)]
            - [event_onset (float)][tab][event_offset (float)]
            - [event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
            - [file(string)][event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
            - [file(string)[tab][scene_label(string)][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)]
            - [file(string)[tab][scene_label(string)][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)][tab][source(single character)]
            - [file(string)[tab][scene_label(string)][tab][event_onset (float)][tab][event_offset (float)][tab][event_label (string)][tab][source(string)]
            - [file(string)[tab][tags (list of strings, delimited with ;)]
            - [file(string)[tab][scene_label(string)][tab][tags (list of strings, delimited with ;)]
            - [file(string)[tab][scene_label(string)][tab][tags (list of strings, delimited with ;)][tab][event_onset (float)][tab][event_offset (float)]

        Parameters
        ----------
        filename : str
            Path to the event list in text format (csv). If none given, one given for class constructor is used.
            Default value "None"

        Returns
        -------
        data : list of event dicts
            List containing event dicts

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        if not os.path.isfile(self.filename):
            raise IOError('{0}: File not found [{1}]'.format(self.__class__.__name__, self.filename))

        data = []
        field_validator = FieldValidator()
        with open(self.filename, 'rtU') as f:
            for row in csv.reader(f, delimiter=self._delimiter):
                if row:
                    row_format = []
                    for item in row:
                        row_format.append(field_validator.process(item))

                    if row_format == ['audiofile']:
                        # Format: [file]
                        data.append(
                            self.item_class({
                                'file': row[0],
                            })
                        )

                    elif row_format == ['number', 'number']:
                        # Format: [event_onset  event_offset]
                        data.append(
                            self.item_class({
                                'onset': float(row[0]),
                                'offset': float(row[1])
                            })
                        )

                    elif row_format == ['audiofile', 'string']:
                        # Format: [file scene_label]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'string']:
                        # Format: [file scene_label identifier]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'identifier': row[2],
                            })
                        )

                    elif row_format == ['number', 'number', 'string']:
                        # Format: [onset  offset    event_label]
                        data.append(
                            self.item_class({
                                'onset': float(row[0]),
                                'offset': float(row[1]),
                                'event_label': row[2]
                            })
                        )

                    elif row_format == ['audiofile', 'number', 'number', 'string']:
                        # Format: [file onset  offset    event_label]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'onset': float(row[1]),
                                'offset': float(row[2]),
                                'event_label': row[3]
                            })
                        )

                    elif row_format == ['file', 'string', 'number', 'number']:
                        # Format: [file event_label onset  offset]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[1]
                            })
                        )

                    elif row_format == ['audiofile', 'number', 'number', 'string', 'string']:
                        # Format: [file onset  offset    event_label identifier]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'onset': float(row[1]),
                                'offset': float(row[2]),
                                'event_label': row[3],
                                'identifier': row[4],
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'string']:
                        # Format: [file scene_label onset  offset    event_label]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[4]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'string', 'alpha1']:
                        # Format: [file scene_label onset  offset   event_label source_label]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[4],
                                'source_label': row[5]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'string', 'string']:
                        # Format: [file scene_label onset  offset   event_label source_label]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[4],
                                'source_label': row[5]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'string', 'alpha1', 'string']:
                        # Format: [file scene_label onset offset event_label source_label identifier]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[4],
                                'source_label': row[5],
                                'identifier': row[6]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'string', 'string', 'string']:
                        # Format: [file scene_label onset offset event_label source_label identifier]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'event_label': row[4],
                                'source_label': row[5],
                                'identifier': row[6]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'list']:
                        # Format: [file scene_label tags]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'tags': row[2]
                            })
                        )

                    elif row_format == ['audiofile', 'list']:
                        # Format: [file tags]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'tags': row[1]
                            })
                        )

                    elif row_format == ['audiofile', 'string', 'number', 'number', 'list']:
                        # Format: [file scene_label onset offset tags]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'scene_label': row[1],
                                'onset': float(row[2]),
                                'offset': float(row[3]),
                                'tags': row[4]
                            })
                        )

                    else:
                        message = '{0}: Unknown row format [{1}]'.format(self.__class__.__name__, row)
                        logging.getLogger(self.__class__.__name__,).exception(message)
                        raise IOError(message)

        list.__init__(self, data)
        return self

    def save(self, filename=None, delimiter='\t'):
        """Save content to csv file

        Parameters
        ----------
        filename : str
            Filename. If none given, one given for class constructor is used.
            Default value "None"
        delimiter : str
            Delimiter to be used
            Default value "\t"

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename

        f = open(self.filename, 'wt')
        try:
            writer = csv.writer(f, delimiter=delimiter)
            for item in self:
                writer.writerow(item.get_list())
        finally:
            f.close()

        return self

    def event_stat_counts(self):
        """Event count statistics

        Returns
        -------
        dict

        """
        stats = {}
        for event_label in self.unique_event_labels:
            stats[event_label] = len(self.filter(event_label=event_label))
        return stats

    def event_roll(self, label_list=None, time_resolution=0.01, label='event_label'):
        """Event roll

        Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

        Parameters
        ----------
        label_list : list
            List of labels in correct order
        time_resolution : float > 0.0
            Time resolution used when converting event into event roll.
            Default value "0.01"
        label : str
            Meta data field used to create event roll
            Default value "event_label"

        Returns
        -------
        numpy.ndarray [shape=(math.ceil(data_length * 1 / time_resolution), amount of classes)]

        """

        max_offset_value = self.max_offset

        if label_list is None:
            label_list = self.unique_event_labels

        # Initialize event roll
        event_roll = numpy.zeros((int(math.ceil(max_offset_value * 1.0 / time_resolution)), len(label_list)))

        # Fill-in event_roll
        for item in self:
            pos = label_list.index(item[label])

            onset = int(math.floor(item.onset * 1.0 / time_resolution))
            offset = int(math.ceil(item.offset * 1.0 / time_resolution))

            event_roll[onset:offset, pos] = 1

        return event_roll


class EventRoll(object):
    def __init__(self, metadata_container, label_list=None, time_resolution=0.01, label='event_label', length=None):
        """Event roll

        Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

        Parameters
        ----------
        metadata_container : MetaDataContainer
            Meta data
        label_list : list
            List of labels in correct order
        time_resolution : float > 0.0
            Time resolution used when converting event into event roll.
            Default value "0.01"
        label : str
            Meta data field used to create event roll
            Default value "event_label"
        length : int, optional
            length of event roll, if none given max offset of the meta data is used.
            Default value "None"

        """

        self.metadata_container = metadata_container

        if label_list is None:
            self.label_list = metadata_container.unique_event_labels
        else:
            self.label_list = label_list

        self.time_resolution = time_resolution
        self.label = label

        if length is None:
            self.max_offset_value = metadata_container.max_offset
        else:
            self.max_offset_value = length

        # Initialize event roll
        self.event_roll = numpy.zeros(
            (int(math.ceil(self.max_offset_value * 1.0 / self.time_resolution)), len(self.label_list))
        )

        # Fill-in event_roll
        for item in self.metadata_container:
            if item.onset is not None and item.offset is not None:
                if item[self.label]:
                    pos = self.label_list.index(item[self.label])
                    onset = int(numpy.floor(item.onset * 1.0 / self.time_resolution))
                    offset = int(numpy.ceil(item.offset * 1.0 / self.time_resolution))

                    if offset > self.event_roll.shape[0]:
                        # we have event which continues beyond max_offset_value
                        offset = self.event_roll.shape[0]

                    if onset <= self.event_roll.shape[0]:
                        # We have event inside roll
                        self.event_roll[onset:offset, pos] = 1

    @property
    def roll(self):
        """Event roll

        Returns
        -------
        event_roll: np.ndarray, shape=(m,k)
            Event roll

        """

        return self.event_roll

    def pad(self, length):
        """Pad event roll's length to given length

        Parameters
        ----------
        length : int
            Length to be padded

        Returns
        -------
        event_roll: np.ndarray, shape=(m,k)
            Padded event roll

        """

        if length > self.event_roll.shape[0]:
            padding = numpy.zeros((length-self.event_roll.shape[0], self.event_roll.shape[1]))
            self.event_roll = numpy.vstack((self.event_roll, padding))

        elif length < self.event_roll.shape[0]:
            self.event_roll = self.event_roll[0:length, :]

        return self.event_roll

    def plot(self):
        """Plot Event roll

        Returns
        -------
        None

        """

        import matplotlib.pyplot as plt
        plt.matshow(self.event_roll.T, cmap=plt.cm.gray, interpolation='nearest', aspect='auto')
        plt.show()


class FieldValidator(object):
    audio_file_extensions = ['wav', 'flac', 'mp3', 'raw']

    def process(self, field):
        if self.is_audiofile(field):
            return 'audiofile'

        elif self.is_number(field):
            return 'number'

        elif self.is_list(field):
            return 'list'

        elif self.is_alpha(field, length=1):
            return 'alpha1'

        elif self.is_alpha(field, length=2):
            return 'alpha2'

        else:
            return 'string'

    def is_number(self, field):
        """Test for number field

        Parameters
        ----------
        field

        Returns
        -------
        bool

        """

        try:
            float(field)  # for int, long and float
        except ValueError:
            try:
                complex(field)  # for complex
            except ValueError:
                return False

        return True

    def is_audiofile(self, field):
        """Test for audio file field

        Parameters
        ----------
        field

        Returns
        -------
        bool

        """

        if field.endswith(tuple(self.audio_file_extensions)):
            return True
        else:
            return False

    def is_list(self, field):
        """Test for list field, valid delimiters [ : ; #]

        Parameters
        ----------
        field

        Returns
        -------
        bool

        """

        if len(re.split(r'[;|:|#"]+', field)) > 1:
            return True
        else:
            return False

    def is_alpha(self, field, length=1):
        """Test for alpha field with length 1

        Parameters
        ----------
        field

        Returns
        -------
        bool

        """
        if len(field) == length and field.isalpha():
            return True
        else:
            return False


class ProbabilityItem(dict):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
            dict

        """

        dict.__init__(self, *args)

        # Process fields
        if 'file' in self:
            # Keep file paths in unix format even under Windows
            self['file'] = posix_path(self['file'])

        if 'timestamp' in self:
            self['timestamp'] = float(self['timestamp'])

        if 'label' in self and self.label:
            self['label'] = self['label'].strip()
            if self['label'].lower() == 'none':
                self['label'] = None

        if 'probability' in self:
            self['probability'] = float(self['probability'])

    def __str__(self):
        if len(self.file) > 40:
            file_string = '...'+self.file[-37:]
        else:
            file_string = self.file

        string_data = '  {0:<40s} |'.format(
            file_string if file_string is not None else '---'
        )

        if self.timestamp is not None:
            string_data += ' {:10.8f} |'.format(self.timestamp)
        else:
            string_data += ' {:>10s} |'.format('---')

        string_data += ' {:<22s} |'.format(self.label if self.label is not None else '---')

        if self.probability is not None:
            string_data += ' {:18.8f} |'.format(self.probability)
        else:
            string_data += ' {:>18s} |'.format('---')

        return string_data

    @staticmethod
    def get_header():
        string_data = '  {0:<40s} | {1:<10s} | {2:<22s} | {3:<18s} |\n'.format(
            'File',
            'Timestamp',
            'Label',
            'Probability'
        )

        string_data += '  {0:<40s} + {1:<10s} + {2:<22s} + {3:<18s} +\n'.format(
            '-' * 40,
            '-' * 10,
            '-' * 22,
            '-' * 18
        )

        return string_data

    @property
    def file(self):
        """Filename

        Returns
        -------
        str or None
            filename

        """

        if 'file' in self:
            return self['file']
        else:
            return None

    @file.setter
    def file(self, value):
        # Keep file paths in unix format even under Windows
        self['file'] = posix_path(value)

    @property
    def label(self):
        """Label

        Returns
        -------
        str or None
            label

        """

        if 'label' in self:
            return self['label']
        else:
            return None

    @label.setter
    def label(self, value):
        self['label'] = value

    @property
    def timestamp(self):
        """timestamp

        Returns
        -------
        float or None
            timestamp

        """

        if 'timestamp' in self:
            return self['timestamp']
        else:
            return None

    @timestamp.setter
    def timestamp(self, value):
        self['timestamp'] = float(value)

    @property
    def probability(self):
        """probability

        Returns
        -------
        float or None
            probability

        """

        if 'probability' in self:
            return self['probability']
        else:
            return None

    @probability.setter
    def probability(self, value):
        self['probability'] = float(value)

    @property
    def id(self):
        """Unique item identifier

        ID is formed by taking MD5 hash of the item data.

        Returns
        -------
        id : str
            Unique item id
        """

        string = ''
        if self.file:
            string += self.file
        if self.timestamp:
            string += '{:8.4f}'.format(self.timestamp)
        if self.label:
            string += self.label
        if self.probability:
            string += '{:8.4f}'.format(self.probability)

        return get_parameter_hash(string)

    def get_list(self):
        """Return item values in a list with specified order.

        Returns
        -------
        list
        """
        fields = list(self.keys())

        # Select only valid fields
        valid_fields = ['file', 'label', 'probability', 'timestamp']
        fields = list(set(fields).intersection(valid_fields))
        fields.sort()

        if fields == ['file', 'label', 'probability']:
            return [self.file, self.label, self.probability]

        elif fields == ['file', 'label', 'probability', 'timestamp']:
            return [self.file, self.timestamp, self.label, self.probability]

        else:
            message = '{name}: Invalid meta data format [{format}]'.format(
                name=self.__class__.__name__,
                format=str(fields)
            )
            raise ValueError(message)


class ProbabilityContainer(ListFile, MetaMixin):
    valid_formats = ['csv', 'txt']

    def __init__(self, *args, **kwargs):
        super(ProbabilityContainer, self).__init__(*args, **kwargs)
        self.item_class = ProbabilityItem

        # Convert all items in the list to ProbabilityItem
        for item_id in range(0, len(self)):
            if not isinstance(self[item_id], self.item_class):
                self[item_id] = self.item_class(self[item_id])

    def __add__(self, other):
        return self.update(super(ProbabilityContainer, self).__add__(other))

    @property
    def file_list(self):
        """List of unique files in the container

        Returns
        -------
        list

        """

        files = {}
        for item in self:
            files[item.file] = item.file

        return sorted(files.values())

    @property
    def unique_labels(self):
        """Get unique labels

        Returns
        -------
        labels: list, shape=(n,)
            Unique labels in alphabetical order

        """

        labels = []
        for item in self:
            if 'label' in item and item['label'] not in labels:
                labels.append(item.label)

        labels.sort()
        return labels

    def filter(self, filename=None, file_list=None, label=None):
        """Filter content

        Parameters
        ----------
        filename : str, optional
            Filename to be matched
            Default value "None"
        file_list : list, optional
            List of filenames to be matched
            Default value "None"
        label : str, optional
            Label to be matched
            Default value "None"

        Returns
        -------
        ProbabilityContainer

        """

        data = []
        for item in self:
            matched = False
            if filename and item.file == filename:
                matched = True
            if file_list and item.file in file_list:
                matched = True
            if label and item.label == label:
                matched = True

            if matched:
                data.append(copy.deepcopy(item))

        return ProbabilityContainer(data)

    def get_string(self):
        """Get content in string format

        Parameters
        ----------

        Returns
        -------
        str
            Multi-line string

        """

        string_data = ''
        string_data += self.item_class().get_header()
        for filename in self.file_list:
            for i in self.filter(filename=filename):
                string_data += str(self.item_class(i)) + '\n'
            string_data += '\n'

        return string_data

    def load(self, filename=None):
        """Load probability list from delimited text file (csv-formated)

        Preferred delimiter is tab, however, other delimiters are supported automatically (they are sniffed automatically).

        Supported input formats:
            - [file(string)][label(string)][probability(float)]

        Parameters
        ----------
        filename : str
            Path to the probability list in text format (csv). If none given, one given for class constructor is used.
            Default value "None"

        Returns
        -------
        data : list of probability item dicts
            List containing probability item dicts

        """

        if filename:
            self.filename = filename
            self.format = self.detect_file_format(self.filename)

        if not os.path.isfile(self.filename):
            raise IOError('{0}: File not found [{1}]'.format(self.__class__.__name__, self.filename))

        data = []
        field_validator = FieldValidator()

        with open(self.filename, 'rt') as f:
            for row in csv.reader(f, delimiter=self._delimiter):
                if row:
                    row_format = []
                    for item in row:
                        row_format.append(field_validator.process(item))

                    if row_format == ['audiofile', 'string', 'number']:
                        # Format: [file label probability]
                        data.append(
                            self.item_class({
                                'file': row[0],
                                'label': row[1],
                                'probability': row[2],
                            })
                        )

                    else:
                        message = '{0}: Unknown row format [{1}]'.format(self.__class__.__name__, row)
                        logging.getLogger(self.__class__.__name__,).exception(message)
                        raise IOError(message)

        list.__init__(self, data)
        return self

    def save(self, filename=None, delimiter='\t'):
        """Save content to csv file

        Parameters
        ----------
        filename : str
            Filename. If none given, one given for class constructor is used.
            Default value "None"
        delimiter : str
            Delimiter to be used
            Default value "\t"

        Returns
        -------
        self

        """

        if filename:
            self.filename = filename

        f = open(self.filename, 'wt')
        try:
            writer = csv.writer(f, delimiter=delimiter)
            for item in self:
                writer.writerow(item.get_list())
        finally:
            f.close()

        return self

