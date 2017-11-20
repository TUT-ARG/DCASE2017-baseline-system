#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameters
==========
Class for parameter handling. ParameterContainer is based on dict and supports reading and writing to YAML files.

Example YAML file:

.. code-block:: yaml

    active_set: SET1

    sets:
        - set_id: SET1
          processor1:
            method: special_method_2
        - set_id: SET2
          processor1:
            method: special_method_3

    defaults:
        flow:
            task1: true
            task2: true
            task3: true

        processor1:
            method: special_method_1
            field1: 44100
            field2: 22050

        processor1_method_parameters:
            special_method_1:
                field1: 'test1'

            special_method_2:
                field1: 'test2'

            special_method_3:
                field1: 'test3'

        processor2:
            recipe: special_method_1;special_method_2;special_method_3

        processor2_method_parameters:
            special_method_1:
                field1: 'test1'

            special_method_2:
                field1: 'test2'

            special_method_3:
                field1: 'test3'

Once :func:`ParameterContainer.process` is called:

1. ``active_set`` field is used to select parameter set to override parameters in the ``defaults`` block. After this parameter container contains only parameters inside defaults block with overrides.
2. Each main level section (``flow``, ``processor``, ``processor_method_parameters`` in the example above) are processed one by one.

    - If section contains ``method``-field, parameters are copied from ``[SECTION_NAME]_method_parameters`` under ``parameters``-field.
    - If section contains ``recipe``-field, recipe is first parsed and parameters are copied from ``[SECTION_NAME]_method_parameters`` under ``parameters``-field.

Parameters after processing:

.. code-block:: yaml

    flow:
        task1: true
        task2: true
        task3: true

    processor1:
        _hash: 1d511b716b3cd075fbc752750b0c5932
        method: special_method_2
        field1: 44100
        field2: 22050
        parameters:
            field1: 'test2'

    processor2:
        _hash: f17897bd2a133d1c1d1c853e491d2a3a
        recipe:
            - method: special_method_1
            - method: special_method_2
            - method: special_method_3

        special_method_1;special_method_2;special_method_3
        parameters:
            special_method_1:
                field1: 'test1'

            special_method_2:
                field1: 'test2'

            special_method_3:
                field1: 'test3'

Recipe
^^^^^^

Recipe special field can be used to select multiple methods. It is specially useful for constructing feature matrix from multiple sources.
Method blocks in the string are delimited with ``;`` (e.g. method1;method2;method1).

Individual items in this list can be formatted following way:

- [method_name (string)]                                                       => full vector
- [method_name (string)]=[start index (int)]-[end index (int)]                 => default channel 0 and vector [start:end]
- [method_name (string)]=[channel (int)]:[start index (int)]-[end index (int)] => specified channel and vector [start:end]
- [method_name (string)]=1,2,3,4,5                                             => vector [1,2,3,4,4]
- [method_name (string)]=0                                                     => specified channel and full vector


Paths and parameter hash
^^^^^^^^^^^^^^^^^^^^^^^^

Parameters under each section is used to form parameter hash. ParameterContainer's property
:py:attr:`dcase_framework.parameters.ParameterContainer.path_structure` defines how these section wise parameter hashes
are used to form storage paths for each section. The main idea is that when parameters change path will change and when
the parameters are the same path is the same allowing reusing already stored data (process with correct parameters).

**Path structure**

Example definition for path structure.

.. code-block:: python

    self.path_structure = {
        'feature_extractor': [
            'feature_extractor.parameters.*'
        ],
        'feature_normalizer': [
            'feature_extractor.parameters.*'
        ],
        'learner': [
            'feature_extractor',
            'feature_normalizer',
            'feature_aggregator',
            'learner'
        ],
        'recognizer': [
            'feature_extractor',
            'feature_normalizer',
            'feature_aggregator',
            'learner',
            'recognizer'
        ],
        'evaluator': [
        ]
    }

One can use wild card for lists (e.g. ``feature_extractor.parameters.*``), in this case each item in the list is producing individual path. This can be used to
make paths, for examples, for each feature extractor separately.


This will lead following paths:

.. code-block:: txt

    feature_extractor/feature_extractor_68a40f5e3b77df9564aaa68c92e95be9/
    feature_extractor/feature_extractor_74c5e3ce692f5973c5071c1cf0a89ee0/
    feature_extractor/feature_extractor_661304966061610bc09744166b10f76e/

    feature_normalizer/feature_extractor_68a40f5e3b77df9564aaa68c92e95be9/
    feature_normalizer/feature_extractor_74c5e3ce692f5973c5071c1cf0a89ee0/
    feature_normalizer/feature_extractor_661304966061610bc09744166b10f76e/

    learner/feature_extractor_5ca1f32c65b3eea59e1bb27b09b747ea/feature_normalizer_67b9b20ff555e8eaee22f5e50695df8b/feature_aggregator_baaf606d9ac1eaca43a6a24b599998a9/learner_624a422b47a32e20b90ad6e6151057f8

    recognizer/feature_extractor_5ca1f32c65b3eea59e1bb27b09b747ea/feature_normalizer_67b9b20ff555e8eaee22f5e50695df8b/feature_aggregator_baaf606d9ac1eaca43a6a24b599998a9/learner_624a422b47a32e20b90ad6e6151057f8/recognizer_08c503973f61ef4c4c5f7c56709d801c

Parameter section used to form hash will be saved in each sub folder (parameters.yaml) to make it easier handle files manually if needed.

**Hash**

Parameter hash value is md5 hash for stringified parameter dict of the section, with a few clean ups helping to keep
hash compatible when extending parameter selection in the section later. Following rules are used:

- If section contains field ``enable`` with value ``False`` all other fields inside this section are excluded from the parameter hash calculation. This will make the hash robust if section is not used but still unused parameters are changed.
- If section contains fields with value ``False``, this field is excluded from the parameter hash calculation. This will enable to add new flag parameters, without changing hash, just define the new flag so that previous behaviour is happening when this field is set to false.
- If section contains any of the ``non_hashable_fields`` fields, those are excluded from the parameter hash calculation. These fields are set when :class:`ParameterContainer` is constructed, and they usually are fields used to print various values to the console. These fields do not change the system output to be saved onto disk, and hence they are excluded from hash.

Use :py:attr:`dcase_framework.parameters.ParameterContainer.non_hashable_fields` to exclude fields from hash. use  :py:attr:`dcase_framework.parameters.ParameterContainer.control_sections` to omit hash calculation for parameter
sections which do not needed it.

ParameterContainer
^^^^^^^^^^^^^^^^^^

Usage examples:

.. code-block:: python
    :linenos:

    # Load parameters
    params = ParameterContainer().load(filename='parameters.yaml')
    # Process parameters
    params.process()
    # Print parameters
    print(params)
    # Get parameters
    value = get_path('section1.parameter1')

.. autosummary::
    :toctree: generated/

    ParameterContainer
    ParameterContainer.load
    ParameterContainer.save
    ParameterContainer.exists
    ParameterContainer.get_path
    ParameterContainer.show
    ParameterContainer.log
    ParameterContainer.override
    ParameterContainer.process
    ParameterContainer.process_method_parameters
    ParameterContainer.get_hash

"""

from __future__ import print_function, absolute_import
from six import iteritems

import os
import hashlib
import json
import copy
import numpy
import itertools
import platform

from .files import ParameterFile
from .containers import ContainerMixin, DottedDict


class ParameterContainer(ParameterFile, ContainerMixin):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        project_base : str
            Absolute path to the project root
        section_process_order : list, optional
            Parameter section processing order. Given dict is used to override internal default list.
            Default value "None"
        path_structure : dict of lists, optional
            Defines how paths are created, section hash is used to create unique folder names. Given dict is used to
            override internal default list.
            Default value "None"
        method_dependencies : dict of dicts, optional
            Given dict is used to override internal default list.
            Default value "None"
        magic_field : dict, optional
            Dict of field names for specific tasks. Given dict is used to override internal default list.
            Default value "None"
        non_hashable_fields : list, optional
            List of fields skipped when parameter hash for the section is calculated. Given list is used to override
            internal default list.
            Default value "None"
        control_sections : list, optional
            List of top level sections used for framework control, for these section no hash is calculated. Given list
            is used to override internal default list.
            Default value "None"
        """

        super(ParameterContainer, self).__init__(*args, **kwargs)

        # Mark container non-processed
        self.processed = False

        # Project base path
        if kwargs.get('project_base'):
            self.project_base = kwargs.get('project_base')
        else:
            self.project_base = os.path.dirname(os.path.realpath(__file__))
            if os.path.split(self.project_base)[1] == 'src':
                # If we are in 'src' folder remove one level
                self.project_base = os.path.join(os.path.split(self.project_base)[0])

        # Define section processing order
        self.section_process_order = [
            'flow',
            'general',
            'logging',
            'path',
            'dataset',
            'dataset_method_parameters',
            'feature_extractor',
            'feature_extractor_method_parameters',
            'feature_stacker',
            'feature_stacker_method_parameters',
            'feature_normalizer',
            'feature_normalizer_parameters',
            'feature_aggregator',
            'feature_aggregator_parameters',
            'learner',
            'recognizer',
            'learner_method_parameters',
            'recognizer_method_parameters',
            'evaluator',
        ]
        if kwargs.get('section_process_order'):
            self.path_structure.update(kwargs.get('section_process_order'))

        # Define how paths are constructed from section hashes
        self.path_structure = {
            'feature_extractor': [
                'feature_extractor.parameters.*'
            ],
            'feature_normalizer': [
                'feature_extractor.parameters.*'
            ],
            'learner': [
                'feature_extractor',
                'feature_stacker',
                'feature_normalizer',
                'feature_aggregator',
                'learner'
            ],
            'recognizer': [
                'feature_extractor',
                'feature_stacker',
                'feature_normalizer',
                'feature_aggregator',
                'learner',
                'recognizer'
            ],
            'evaluator': [
            ]
        }
        if kwargs.get('path_structure'):
            self.path_structure.update(kwargs.get('path_structure'))

        # Method dependencies map
        self.method_dependencies = {
            'feature_extractor': {
                'mel': None,
                'mfcc': None,
                'mfcc_delta': 'feature_extractor.mfcc',
                'mfcc_acceleration': 'feature_extractor.mfcc',
            },
        }
        if kwargs.get('method_dependencies'):
            self.method_dependencies.update(kwargs.get('method_dependencies'))

        # Map for magic field names
        self.magic_field = {
            'default-parameters': 'defaults',
            'set-list': 'sets',
            'set-id': 'set_id',
            'active-set': 'active_set',
            'parameters': 'parameters',
            'method': 'method',
            'recipe': 'recipe',
            'path': 'path',
            'flow': 'flow',
            'logging': 'logging',
            'general': 'general',
            'evaluator': 'evaluator',
        }
        if kwargs.get('magic_field'):
            self.magic_field.update(kwargs.get('magic_field'))

        # Fields to be skipped when parameter hash is calculated
        self.non_hashable_fields = [
            '_hash',
            'verbose',
            'print_system_progress',
            'log_system_parameters',
            'log_system_progress',
            'log_learner_status',
            'show_model_information',
            'use_ascii_progress_bar',
            'label',
            'active_scenes',
            'active_events',
            'plotting_rate',
            'focus_span',
            'output_format',
        ]
        if kwargs.get('non_hashable_fields'):
            self.non_hashable_fields.update(kwargs.get('non_hashable_fields'))

        # Parameters sections which will not be included in the master parameter hash
        self.control_sections = [
            'flow',
            'path',
            'logging',
            'general',
            'evaluator',
        ]
        if kwargs.get('control_sections'):
            self.control_sections.update(kwargs.get('control_sections'))

    def override(self, override):
        """Override container content recursively.

        Parameters
        ----------
        override : dict, str
            Depending type following is done:

            - If dict given, this is used directly to override parameters in the container.
            - If str is given which is a filename of existing file on disk, parameter file is loaded and it is used to override container parameters
            - If str is given which contains JSON formatted parameters, content is used to override container parameters

        Raises
        ------
        ImportError:
            JSON import failed
        ValueError:
            Not JSON formatted string given

        Returns
        -------
        self

        """

        if isinstance(override, dict):
            self.merge(override=override)
        elif isinstance(override, str) and os.path.isfile(override):
            self.merge(override=ParameterFile(filename=override).load())
        elif isinstance(override, str):
            try:
                try:
                    import ujson as json
                except ImportError:
                    try:
                        import json
                    except ImportError:
                        message = '{name}: Unable to import json module'.format(
                            name=self.__class__.__name__
                        )

                        self.logger.exception(message)
                        raise ImportError(message)

                self.merge(override=json.loads(override))

            except:
                message = '{name}: Not JSON formatted string given'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise ValueError(message)

        return self

    def process(self, create_directories=True, create_parameter_hints=True):
        """Process parameters

        Parameters
        ----------
        create_directories : bool
            Create directories
            Default value "True"
        create_parameter_hints : bool
            Create parameters files to all data folders
            Default value "True"

        Raises
        ------
        ValueError:
            No valid active set given

        Returns
        -------
        self

        """

        if len(self) == 0:
            message = '{name}: Parameter container empty'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise IOError(message)

        if not self.processed:
            for section_id, section in enumerate(self.control_sections):
                if section in self.magic_field:
                    self.control_sections[section_id] = self.magic_field[section]

            if (self.magic_field['default-parameters'] in self and
               self.magic_field['set-list'] in self and
               self.magic_field['active-set'] in self):

                default_params = copy.deepcopy(self[self.magic_field['default-parameters']])
                active_set_id = self[self.magic_field['active-set']]

                override_params = copy.deepcopy(
                    self._search_list_of_dictionaries(
                        key=self.magic_field['set-id'],
                        value=active_set_id,
                        list_of_dictionaries=self[self.magic_field['set-list']]
                    )
                )

                if not override_params:
                    message = '{name}: No valid active set given [{set_name}]'.format(
                        name=self.__class__.__name__,
                        set_name=active_set_id
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

                dict.clear(self)                            # Empty current content
                dict.update(self, default_params)           # Insert default parameters
                self['active_set'] = active_set_id

                self.merge(override=override_params)        # Merge override parameters into default parameters

            if self.magic_field['default-parameters'] in self:
                default_params = copy.deepcopy(self[self.magic_field['default-parameters']])
                dict.clear(self)
                dict.update(self, default_params)

            # Get processing order for sections
            section_list = []
            for section in self.section_process_order + list(set(list(self.keys())) - set(self.section_process_order)):
                if section in self:
                    section_list.append(section)

            # Parameter processing starts
            self._convert_main_level_to_dotted()

            self._preprocess_paths()

            # 1. Process parameters
            for section in section_list:
                field_process_func = getattr(self, '_process_{}'.format(section), None)
                if field_process_func is not None:
                    field_process_func()

                if (self[section] and
                   (self.magic_field['method'] in self[section] or self.magic_field['recipe'] in self[section]) and
                    section+'_method_parameters' in self):

                    field_process_parameters_func = getattr(self, '_process_{}_method_parameters'.format(section), None)
                    if field_process_parameters_func is not None:
                        field_process_parameters_func()

            self._add_hash_to_method_parameters()

            # 2. Methods and recipies
            for section in section_list:
                self.process_method_parameters(section=section)

            # 3. Inject dependencies
            for section in section_list:
                if isinstance(self[section], dict) and self[section] and self.magic_field['parameters'] in self[section]:
                    for key, item in iteritems(self[section][self.magic_field['parameters']]):
                        if (section in self.method_dependencies and
                            key in self.method_dependencies[section] and self.method_dependencies[section][key]):

                            fields = self.method_dependencies[section][key].split('.')
                            if len(fields) == 1:
                                item['dependency_parameters'] = copy.deepcopy(
                                    self[section + '_method_parameters'][self.method_dependencies[section][key]]
                                )

                                item['dependency_method'] = self.method_dependencies[section][key]

                            elif len(fields) == 2:
                                item['dependency_parameters'] = copy.deepcopy(
                                    self[fields[0] + '_method_parameters'][fields[1]]
                                )

                                item['dependency_method'] = fields[1]

            # 4. Add hash
            self._add_hash_to_main_parameters()
            self._add_main_hash()

            # 5. Post process paths
            self._postprocess_paths(
                create_directories=create_directories,
                create_parameter_hints=create_parameter_hints
            )

            self.processed = True

            # 6. Clean up
            # self._clean_unused_parameters()

        return self

    def process_method_parameters(self, section):
        """Process methods and recipes in the section

        Processing rules for fields:

        - "method" => search for parameters from [section]_method_parameters -section
        - "recipe" => parse recipe and search for parameters from [section]_method_parameters -section
        - "\*recipe" => parse recipe

        Parameters
        ----------
        section : str
            Section name

        Raises
        ------
        ValueError:
            Invalid method for parameter field


        Returns
        -------
            self

        """

        # Inject method parameters
        if self[section]:
            if self.magic_field['method'] in self[section]:
                if (section + '_method_parameters' in self and
                   self[section][self.magic_field['method']] in self[section + '_method_parameters']):

                    self[section]['parameters'] = copy.deepcopy(
                        self[section + '_method_parameters'][self[section][self.magic_field['method']]]
                    )

                else:
                    message = '{name}: Invalid method for parameter field, {field}->method={method}'.format(
                        name=self.__class__.__name__,
                        field=section,
                        method=self[section][self.magic_field['method']]
                    )

                    self.logger.exception(message)
                    raise ValueError(message)

            # parse recipes
            for field in self[section]:
                if field.endswith(self.magic_field['recipe']):
                    self[section][field] = self._parse_recipe(recipe=self[section][field])

            # Inject recipes
            if self.magic_field['recipe'] in self[section]:
                self[section][self.magic_field['parameters']] = {}
                for item in self[section][self.magic_field['recipe']]:
                    if (section + '_method_parameters' in self and
                       item[self.magic_field['method']] in self[section + '_method_parameters']):

                        self[section]['parameters'][item[self.magic_field['method']]] = copy.deepcopy(
                            self[section + '_method_parameters'][item[self.magic_field['method']]]
                        )

                    else:
                        message = '{name}: Cannot find any parameters for the method in the recipe field, {field}->recipe={method}'.format(
                            name=self.__class__.__name__,
                            field=section,
                            method=item[self.magic_field['method']]
                        )

                        self.logger.exception(message)
                        raise ValueError(message)

        return self

    @staticmethod
    def _check_paths(paths, create=True):
        def make_path(path):
            if isinstance(path, str) and not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except OSError as exception:
                    pass
        if create:
            if isinstance(paths, str):
                make_path(paths)

            elif isinstance(paths, dict):
                for key, value in iteritems(paths):
                    make_path(value)

            elif isinstance(paths, list):
                for value in paths:
                    make_path(value)

    def _preprocess_paths(self):
        # Translate separators if in Windows
        if platform.system() == 'Windows':
            for path_field in self['path']:
                self['path'][path_field] = self['path'][path_field].replace('/', os.path.sep)

        # If given path is relative, make it absolute
        if not os.path.isabs(self.get_path('path.data')):
            self['path']['data'] = os.path.join(self.project_base,
                                                self.get_path('path.data'))

        if not os.path.isabs(self.get_path('path.system_base')):
            self['path']['system_base'] = os.path.join(self.project_base,
                                                       self.get_path('path.system_base'))

        if not os.path.isabs(self.get_path('path.recognizer_challenge_output')):
            self['path']['recognizer_challenge_output'] = os.path.join(
                self.project_base,
                self.get_path('path.recognizer_challenge_output')
            )

        if not os.path.isabs(self.get_path('path.logs')):
            self['path']['logs'] = os.path.join(self.project_base,
                                                self.get_path('path.logs'))

    def _postprocess_paths(self, create_directories=True, create_parameter_hints=True):
        # Make sure extended paths exists before saving parameters in them

        # Check main paths
        if create_directories:
            if 'data' in self['path']:
                self._check_paths(paths=self['path']['data'])
            if 'system_base' in self['path']:
                self._check_paths(paths=self['path']['system_base'])
            if 'logs' in self['path']:
                self._check_paths(paths=self['path']['logs'])
            if 'recognizer_challenge_output' in self['path']:
                self._check_paths(paths=self['path']['recognizer_challenge_output'])

        # Check path_structure
        for field, structure in iteritems(self.path_structure):
            path = self._get_extended_path(path_label=field, structure=structure)
            if create_directories:
                self._check_paths(paths=path)
            if create_parameter_hints:
                self._save_path_parameters(
                    base=[os.path.join(self['path']['system_base'], self['path'][field])],
                    structure=structure
                )

            self['path'][field] = path

    @staticmethod
    def _join_paths(path_parts):
        if len(path_parts) > 1:
            for i, value in enumerate(path_parts):
                if isinstance(value, str):
                    path_parts[i] = [value]

            if len(path_parts) == 2:
                path_parts = list(itertools.product(path_parts[0], path_parts[1]))
            elif len(path_parts) == 3:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2]))
            elif len(path_parts) == 4:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2], path_parts[3]))
            elif len(path_parts) == 5:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2], path_parts[3],
                                                    path_parts[4]))
            elif len(path_parts) == 6:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2], path_parts[3],
                                                    path_parts[4], path_parts[5]))
            elif len(path_parts) == 7:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2], path_parts[3],
                                                    path_parts[4], path_parts[5], path_parts[6]))
            elif len(path_parts) == 8:
                path_parts = list(itertools.product(path_parts[0], path_parts[1], path_parts[2], path_parts[3],
                                                    path_parts[4], path_parts[5], path_parts[6], path_parts[7]))
        out_path = []
        for l in path_parts:
            out_path.append(os.path.join(*l))
        return out_path

    def _get_extended_path(self, path_label, structure):
        path_parts = [os.path.join(self['path']['system_base'], self['path'][path_label])]
        if structure:
            keys = []
            wild_card_found = False
            for part in structure:
                if '*' in part:
                    wild_card_found = True
                    path_ = self.get_path(data=self, dotted_path=part[:part.find('*') - 1])
                    if path_:
                        keys = path_.keys()

                param_hash = self.get_path(data=self, dotted_path=part + '._hash')
                if param_hash is not None:
                    if isinstance(param_hash, list):
                        directory_name = []
                        for h in param_hash:
                            directory_name.append(part.split('.')[0]+'_'+h)
                    else:
                        directory_name = self._get_directory_name(prefix=part.split('.')[0], param_hash=param_hash)

                    path_parts.append(directory_name)

            paths = self._join_paths(path_parts)

            if not wild_card_found and len(paths) == 1:
                return paths[0]
            else:
                return dict(zip(keys, paths))
        else:
            return os.path.join(self['path']['system_base'], self['path'][path_label])

    def _get_directory_name(self, prefix, param_hash):
        if platform.system() == 'Windows':
            # Use short directory names and truncated hash for Windows, as it has path length limit (260)
            return param_hash[0:20]

        else:
            return prefix + '_' + param_hash

    def _save_path_parameters(self, base, structure, parameter_filename='parameters.yaml'):
        path_parts = [os.path.join(base[0])]
        for part in structure:
            param_hash = self.get_path(data=self, dotted_path=part + '._hash')
            if param_hash is not None:
                if isinstance(param_hash, list):
                    directory_name = []
                    for h in param_hash:
                        directory_name.append(part.split('.')[0] + '_' + h)
                else:
                    directory_name = self._get_directory_name(prefix=part.split('.')[0], param_hash=param_hash)

                parameters = self.get_path(data=self, dotted_path=part)
                path_parts.append(directory_name)

                current_path = self._join_paths(path_parts)

                if isinstance(current_path, str):
                    ParameterContainer(parameters).save(filename=os.path.join(current_path[0], parameter_filename))
                else:
                    if isinstance(parameters, dict):
                        ParameterContainer(parameters).save(filename=os.path.join(current_path[0], parameter_filename))
                    else:
                        for path_id, path in enumerate(current_path):
                            if parameters[path_id]:
                                ParameterContainer(parameters[path_id]).save(filename=os.path.join(path, parameter_filename))

    def _save_path_parameters_all(self):
        for path_label, structure in iteritems(self.path_structure):
            path_parts = [os.path.join(self['path']['system_base'], self['path'][path_label])]
            for part in structure:
                param_hash = self.get_path(data=self, dotted_path=part + '.hash')
                parameters = self.get_path(data=self, dotted_path=part)
                path_parts.append(param_hash)

                current_path = self._join_paths(path_parts)

                if len(current_path) == 1:
                    ParameterContainer(parameters).save(filename=os.path.join(current_path[0], 'parameters.yaml'))
                else:
                    for path_id, path in enumerate(current_path):
                        if parameters[path_id]:
                            ParameterContainer(parameters[path_id]).save(filename=os.path.join(path, 'parameters.yaml'))

    def _add_hash_to_main_parameters(self):
        for field, params in iteritems(self):
            if isinstance(params, dict):
                if field not in self.control_sections and self[field]:
                    self[field]['_hash'] = self.get_hash(data=self[field])

    def _add_hash_to_method_parameters(self):
        for field in self:
            if field.endswith('_method_parameters'):
                for key, params in iteritems(self[field]):
                    if params:
                        params['_hash'] = self.get_hash(data=params)

    def _add_main_hash(self):
        data = {}
        for field, params in iteritems(self):
            if isinstance(params, dict):
                if field not in self.control_sections and self[field]:
                    data[field] = self.get_hash(data=self[field])
        self['_hash'] = self.get_hash(data=data)

    @staticmethod
    def _parse_recipe(recipe):
        """Parse feature vector recipe

        Overall format: [block #1];[block #2];[block #3];...

        Block formats:
         - [extractor (string)] = full vector
         - [extractor (string)]=[start index (int)]-[end index (int)] => default channel 0 and vector [start:end]
         - [extractor (string)]=[channel (int)]:[start index (int)]-[end index (int)] => specified channel and vector [start:end]
         - [extractor (string)]=1,2,3,4,5 => vector [1,2,3,4,4]
         - [extractor (string)]=0 => specified channel and full vector

        Parameters
        ----------
        recipe : str
            Feature recipe

        Returns
        -------
        data : dict
            Feature recipe structure

        """

        # Define delimiters
        delimiters = {
            'block': ';',
            'detail': '=',
            'dimension': ':',
            'segment': '-',
            'vector': ','
        }

        data = []
        labels = recipe.split(delimiters['block'])
        for label in labels:
            label = label.strip()
            if label:
                detail_parts = label.split(delimiters['detail'])
                method = detail_parts[0].strip()

                # Default values, used when only extractor is defined e.g. [extractor (string)]; [extractor (string)]
                vector_index_structure = {
                    'channel': 0,
                    'selection': False,
                    'full': True,
                }

                # Inspect recipe further
                if len(detail_parts) == 2:
                    main_index_parts = detail_parts[1].split(delimiters['dimension'])
                    vector_indexing_string = detail_parts[1]

                    if len(main_index_parts) > 1:
                        # Channel has been defined,
                        # e.g. [extractor (string)]=[channel (int)]:[start index (int)]-[end index (int)]
                        vector_index_structure['channel'] = int(main_index_parts[0])
                        vector_indexing_string = main_index_parts[1]

                    vector_indexing = vector_indexing_string.split(delimiters['segment'])
                    if len(vector_indexing) > 1:
                        vector_index_structure['start'] = int(vector_indexing[0].strip())
                        vector_index_structure['end'] = int(vector_indexing[1].strip()) + 1
                        vector_index_structure['full'] = False
                        vector_index_structure['selection'] = False
                    else:
                        vector_indexing = vector_indexing_string.split(delimiters['vector'])
                        if len(vector_indexing) > 1:
                            a = list(map(int, vector_indexing))
                            vector_index_structure['full'] = False
                            vector_index_structure['selection'] = True
                            vector_index_structure['vector'] = a
                        else:
                            vector_index_structure['channel'] = int(vector_indexing[0])
                            vector_index_structure['full'] = True
                            vector_index_structure['selection'] = False

                    current_data = {
                        'method': method,
                        'vector-index': vector_index_structure,
                        # 'parameter-path': 'feature.params.' + extractor
                    }
                else:
                    current_data = {
                        'method': method,
                    }

                data.append(current_data)

        return data

    def _after_load(self, to_return=None):
        self.processed = False

    def _clean_unused_parameters(self):
        for field in list(self.keys()):
            if field.endswith('_method_parameters'):
                del self[field]

    def _convert_main_level_to_dotted(self):
        for key, item in iteritems(self):
            if isinstance(item, dict) and self.magic_field['parameters'] in item:
                item[self.magic_field['parameters']] = DottedDict(item[self.magic_field['parameters']])
            if isinstance(item, dict):
                self[key] = DottedDict(item)

    def _process_logging(self):
        for handler_name, handler_data in iteritems(self['logging']['parameters']['handlers']):
            if 'filename' in handler_data:
                handler_data['filename'] = os.path.join(self['path']['logs'], handler_data['filename'])

    def _process_feature_extractor(self):
        if ('recipe' not in self['feature_extractor'] and 'feature_stacker' in self and
           'stacking_recipe' in self['feature_stacker']):

            self['feature_extractor']['recipe'] = self.get_path('feature_stacker.stacking_recipe')

        if 'win_length_seconds' in self['feature_extractor'] and 'fs' in self['feature_extractor']:
            self['feature_extractor']['win_length_samples'] = int(self.get_path('feature_extractor.win_length_seconds') * self.get_path('feature_extractor.fs'))

        if 'hop_length_seconds' in self['feature_extractor'] and 'fs' in self['feature_extractor']:
            self['feature_extractor']['hop_length_samples'] = int(self.get_path('feature_extractor.hop_length_seconds') * self.get_path('feature_extractor.fs'))

    def _process_feature_normalizer(self):
        if self.get_path('general.scene_handling'):
            self['feature_normalizer']['scene_handling'] = self.get_path('general.scene_handling')

        if self.get_path('general.active_scenes'):
            self['feature_normalizer']['active_scenes'] = self.get_path('general.active_scenes')

        if self.get_path('general.event_handling'):
            self['feature_normalizer']['event_handling'] = self.get_path('general.event_handling')

        if self.get_path('general.active_events'):
            self['feature_normalizer']['active_events'] = self.get_path('general.active_events')

    def _process_feature_extractor_method_parameters(self):
        # Change None feature parameter sections into empty dicts
        for method in list(self['feature_extractor_method_parameters'].keys()):
            if self['feature_extractor_method_parameters'][method] is None:
                self['feature_extractor_method_parameters'][method] = {}

        for method, data in iteritems(self['feature_extractor_method_parameters']):
            data['method'] = method

            # Copy general parameters
            if 'fs' in self['feature_extractor']:
                data['fs'] = self['feature_extractor']['fs']

            if 'win_length_seconds' in self['feature_extractor']:
                data['win_length_seconds'] = self.get_path('feature_extractor.win_length_seconds')
            if 'win_length_samples' in self['feature_extractor']:
                data['win_length_samples'] = self.get_path('feature_extractor.win_length_samples')

            if 'hop_length_seconds' in self['feature_extractor']:
                data['hop_length_seconds'] = self.get_path('feature_extractor.hop_length_seconds')
            if 'hop_length_samples' in self['feature_extractor']:
                data['hop_length_samples'] = self.get_path('feature_extractor.hop_length_samples')

    def _process_feature_aggregator(self):
        if 'win_length_seconds' in self['feature_aggregator'] and 'win_length_seconds' in self['feature_extractor']:
            self['feature_aggregator']['win_length_frames'] = int(numpy.ceil(self.get_path('feature_aggregator.win_length_seconds') / float(self.get_path('feature_extractor.hop_length_seconds'))))

        if 'hop_length_seconds' in self['feature_aggregator'] and 'win_length_seconds' in self['feature_extractor']:
            self['feature_aggregator']['hop_length_frames'] = int(numpy.ceil(self.get_path('feature_aggregator.hop_length_seconds') / float(self.get_path('feature_extractor.hop_length_seconds'))))

    def _process_learner(self):
        win_length_seconds = self.get_path('feature_extractor.win_length_seconds')
        hop_length_seconds = self.get_path('feature_extractor.hop_length_seconds')

        if self.get_path('feature_aggregator.enable'):
            win_length_seconds = self.get_path('feature_aggregator.win_length_seconds')
            hop_length_seconds = self.get_path('feature_aggregator.hop_length_seconds')

        self['learner']['win_length_seconds'] = float(win_length_seconds)
        self['learner']['hop_length_seconds'] = float(hop_length_seconds)

        if self.get_path('general.scene_handling'):
            self['learner']['scene_handling'] = self.get_path('general.scene_handling')

        if self.get_path('general.active_scenes'):
            self['learner']['active_scenes'] = self.get_path('general.active_scenes')

        if self.get_path('general.event_handling'):
            self['learner']['event_handling'] = self.get_path('general.event_handling')

        if self.get_path('general.active_events'):
            self['learner']['active_events'] = self.get_path('general.active_events')

    def _process_learner_method_parameters(self):
        for method, data in iteritems(self['learner_method_parameters']):
            data = DottedDict(data)
            if data.get_path('training.epoch_processing.enable') and not data.get_path('training.epoch_processing.recognizer'):
                data['training']['epoch_processing']['recognizer'] = self.get_path('recognizer')

    def _process_recognizer(self):
        if self.get_path('general.scene_handling'):
            self['recognizer']['scene_handling'] = self.get_path('general.scene_handling')

        if self.get_path('general.active_scenes'):
            self['recognizer']['active_scenes'] = self.get_path('general.active_scenes')

        if self.get_path('general.event_handling'):
            self['recognizer']['event_handling'] = self.get_path('general.event_handling')

        if self.get_path('general.active_events'):
            self['recognizer']['active_events'] = self.get_path('general.active_events')

        if (self.get_path('recognizer.frame_accumulation.enable') and
           self.get_path('recognizer.frame_accumulation.window_length_seconds')):

            self['recognizer']['frame_accumulation']['window_length_frames'] = int(self.get_path('recognizer.frame_accumulation.window_length_seconds')/float(self.get_path('learner.hop_length_seconds')))

        if (self.get_path('recognizer.event_activity_processing.enable') and
           self.get_path('recognizer.event_activity_processing.window_length_seconds')):

            self['recognizer']['event_activity_processing']['window_length_frames'] = int(self.get_path('recognizer.event_activity_processing.window_length_seconds')/float(self.get_path('learner.hop_length_seconds')))

    def _process_evaluator(self):
        if self.get_path('general.scene_handling'):
            self['evaluator']['scene_handling'] = self.get_path('general.scene_handling')

        if self.get_path('general.active_scenes'):
            self['evaluator']['active_scenes'] = self.get_path('general.active_scenes')

        if self.get_path('general.event_handling'):
            self['evaluator']['event_handling'] = self.get_path('general.event_handling')

        if self.get_path('general.active_events'):
            self['evaluator']['active_events'] = self.get_path('general.active_events')

