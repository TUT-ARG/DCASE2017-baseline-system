#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features
==================
Classes for data containers



"""

from __future__ import print_function, absolute_import
from six import iteritems
import numpy
import logging
import hashlib
import json
import copy


class ContainerMixin(object):
    def get_path(self, dotted_path, default=None, data=None):
        """Get value from nested dict with dotted path

        Parameters
        ----------
        dotted_path : str
            String in form of "field1.field2.field3"
        default : str, int, float
            Default value returned if path does not exists
            Default value "None"
        data : dict, optional
            Dict for which path search is done, if None given self is used. Used for recursive path search.
            Default value "None"

        Returns
        -------

        """

        if data is None:
            data = self
        fields = dotted_path.split('.')

        if '*' == fields[0]:
            # Magic field to return all childes in a list
            sub_list = []
            for key, value in iteritems(data):
                if len(fields) > 1:
                    sub_list.append(self.get_path(data=value, dotted_path='.'.join(fields[1:]), default=default))
                else:
                    sub_list.append(value)
            return sub_list
        else:
            if fields[0] in data and len(fields) > 1:
                # Go deeper
                return self.get_path(data=data[fields[0]], dotted_path='.'.join(fields[1:]), default=default)

            elif fields[0] in data and len(fields) == 1:
                # We reached to the node
                return data[fields[0]]

            else:
                return default

    def set_path(self, dotted_path, new_value, data=None):
        """Set value in nested dict with dotted path

        Parameters
        ----------
        dotted_path : str
            String in form of "field1.field2.field3"
        new_value :
            new value to be placed
        data : dict, optional
            Dict for which path search is done, if None given self is used. Used for recursive path search.
            Default value "None"

        Returns
        -------

        """

        if data is None:
            data = self
        fields = dotted_path.split('.')

        if '*' == fields[0]:
            # Magic field to set all childes in a list
            for key, value in iteritems(data):
                if len(fields) > 1:
                    self.set_path(new_value=new_value, data=value, dotted_path='.'.join(fields[1:]))
                else:
                    data[key] = new_value

        else:
            print(fields[0])
            if len(fields) == 1:
                # We reached to the node
                data[fields[0]] = new_value
            else:
                if fields[0] not in data:
                    data[fields[0]] = {}
                elif not isinstance(data[fields[0]], dict):
                    # Overwrite path
                    data[fields[0]] = {}
                self.set_path(new_value=new_value, data=data[fields[0]], dotted_path='.'.join(fields[1:]))

    def _walk(self, d, depth=0):
        """Recursive dict walk to get string of the content nicely formatted

        Parameters
        ----------
        d : dict
            Dict for walking
        depth : int
            Depth of walk, string is indented with this
            Default value 0

        Returns
        -------
            str

        """

        output = ''
        indent = 3
        header_width = 35 - depth*indent

        for k, v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                output += "".ljust(depth * indent)+k+'\n'
                output += self._walk(v, depth + 1)
            else:
                if isinstance(v, numpy.ndarray):
                    # Numpy array or matrix
                    shape = v.shape
                    if len(shape) == 1:
                        output += "".ljust(depth * indent)
                        output += k.ljust(header_width) + " : " + "array (%d)" % (v.shape[0]) + '\n'

                    elif len(shape) == 2:
                        output += "".ljust(depth * indent)
                        output += k.ljust(header_width) + " : " + "matrix (%d,%d)" % (v.shape[0], v.shape[1]) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], str):
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : list (%d)\n" % len(v)
                    for item_id, item in enumerate(v):
                        output += "".ljust((depth + 1) * indent)
                        output += ("["+str(item_id)+"]").ljust(header_width-3) + " : " + str(item) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], numpy.ndarray):
                    # List of arrays
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : list (%d)\n" % len(v)
                    for item_id, item in enumerate(v):
                        if len(item.shape) == 1:
                            output += "".ljust((depth+1) * indent)
                            output += ("["+str(item_id)+"]").ljust(header_width-3) + " : array (%d)" % (item.shape[0]) + '\n'

                        elif len(item.shape) == 2:
                            output += "".ljust((depth+1) * indent)
                            output += ("["+str(item_id)+"]").ljust(header_width-3) + " : matrix (%d,%d)" % (item.shape[0], item.shape[1]) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], dict):
                    output += "".ljust(depth * indent)
                    output += k.ljust(header_width) + " : list (%d)\n" % len(v)

                    for item_id, item in enumerate(v):
                        output += "".ljust((depth + 1) * indent) + "["+str(item_id)+"]" + '\n'
                        output += self._walk(item, depth + 2)

                else:
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : " + str(v) + '\n'

        return output

    def __str__(self):
        return self._walk(self, depth=1)

    def show(self):
        """Print container content

        Returns
        -------
            Nothing

        """

        print(self._walk(self, depth=1))

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

    @staticmethod
    def _search_list_of_dictionaries(key, value, list_of_dictionaries):
        """Search in the list of dictionaries

        Parameters
        ----------
        key : str
            Dict key for the search
        value :
            Value for the key to match
        list_of_dictionaries : list of dicts
            List to search

        Returns
        -------
            Dict or None

        """

        for element in list_of_dictionaries:
            if element.get(key) == value:
                return element
        return None

    def merge(self, override, target=None):
        """ Recursive dict merge

        Parameters
        ----------
        target : dict
            target parameter dict

        override : dict
            override parameter dict

        Returns
        -------
        None

        """

        if not target:
            target = self

        for k, v in iteritems(override):
            if k in target and isinstance(target[k], dict) and isinstance(override[k], dict):
                self.merge(target=target[k], override=override[k])
            else:
                target[k] = override[k]

    def get_hash_for_path(self, dotted_path=None):
        if dotted_path:
            data = self.get_path(dotted_path=dotted_path)
            if data is not None:
                return self.get_hash(data)
            else:
                return None
        else:
            return self.get_hash(self)

    def get_hash(self, data=None):
        """Get unique hash string (md5) for given parameter dict

        Parameters
        ----------
        data : dict
            Input parameters

        Returns
        -------
        md5_hash : str
            Unique hash for parameter dict

        """

        if data is None:
            data = dict(self)

        md5 = hashlib.md5()
        md5.update(str(json.dumps(self._clean_for_hashing(copy.deepcopy(data)), sort_keys=True)).encode('utf-8'))
        return md5.hexdigest()

    def _clean_for_hashing(self, data, non_hashable_fields=None):
        # Recursively remove keys with value set to False, or non hashable fields
        if non_hashable_fields is None and hasattr(self, 'non_hashable_fields'):
            non_hashable_fields = self.non_hashable_fields
        elif non_hashable_fields is None:
            non_hashable_fields = []

        if data:
            if 'enable' in data and not data['enable']:
                return {
                    'enable': False,
                }
            else:
                if isinstance(data, dict):
                    for key in list(data.keys()):
                        value = data[key]
                        if isinstance(value, bool) and value is False:
                            # Remove fields marked False
                            del data[key]
                        elif key in non_hashable_fields:
                            # Remove fields marked in non_hashable_fields list
                            del data[key]
                        elif isinstance(value, dict):
                            if 'enable' in value and not value['enable']:
                                # Remove dict block which is disabled
                                del data[key]
                            else:
                                # Proceed recursively
                                data[key] = self._clean_for_hashing(value)
                    return data
                else:
                    return data
        else:
            return data


class DottedDict(dict, ContainerMixin):
    def __init__(self, *args, **kwargs):
        super(DottedDict, self).__init__(*args, **kwargs)

        self.non_hashable_fields = [
            '_hash',
            'verbose',
        ]
        if kwargs.get('non_hashable_fields'):
            self.non_hashable_fields.update(kwargs.get('non_hashable_fields'))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.__dict__ = state


