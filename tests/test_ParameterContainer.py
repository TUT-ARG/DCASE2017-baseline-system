""" Unit tests for ParameterContainer """

import nose.tools
import sys
sys.path.append('..')
import json
import os
from dcase_framework.parameters import ParameterContainer, ParameterFile
import tempfile


def test_hash():
    data_hash_target = 'be2ebe4cb85c65e7e679c55595c62446'
    data1 = {
        'field1': {
            'enable': True,
            '1': [1, 2, 3],
            '2': 1234,
        },
        'field2': {
            'sub_field1': 1234
        },
        'field3': {
            'enable': False,
            'sub_field1': 1234
        }
    }
    nose.tools.eq_(ParameterContainer(data1).get_hash(), data_hash_target)

    # False valued field
    data2 = {
        'field2': {
            'sub_field1': 1234
        },
        'field1': {
            '2': 1234,
            '1': [1, 2, 3],
            'enable': True,
        },
        'field3': {
            'enable': False,
            'sub_field1': 1234
        }
    }
    nose.tools.eq_(ParameterContainer(data2).get_hash(), data_hash_target)

    # False valued field
    data3 = {
        'field1': {
            'enable': True,
            '1': [1, 2, 3],
            '2': 1234,
        },
        'field2': {
            'sub_field1': 1234,
            'field': False
        },
        'field3': {
            'enable': False,
            'sub_field1': 1234
        }
    }
    nose.tools.eq_(ParameterContainer(data3).get_hash(), data_hash_target)

    # Change value in disabled section
    data4 = {
        'field1': {
            'enable': True,
            '1': [1, 2, 3],
            '2': 1234,
        },
        'field2': {
            'sub_field1': 1234,
            'field': False
        },
        'field3': {
            'enable': False,
            'sub_field1': 4321
        }
    }
    nose.tools.eq_(ParameterContainer(data4).get_hash(), data_hash_target)

def test_path():
    params = ParameterContainer().load(filename=os.path.join('material', 'test.parameters.yaml'))
    params.process(create_directories=False, create_parameter_hints=False)

    nose.tools.assert_list_equal(sorted(list(params.get_path('path.feature_extractor').keys())), ['featA', 'featB'])
    nose.tools.assert_list_equal(sorted(list(params.get_path('path.feature_normalizer').keys())), ['featA', 'featB'])

    nose.tools.eq_(len(params.get_path('path.learner').replace(params.get_path('path.system_base'), '').split('/')),
                   len(params.path_structure['learner'])+1)
    nose.tools.eq_(len(params.get_path('path.recognizer').replace(params.get_path('path.system_base'), '').split('/')),
                   len(params.path_structure['recognizer'])+1)


def test_active_set_selection():
    params = ParameterContainer().load(filename=os.path.join('material', 'test.parameters.yaml'))
    params['active_set'] = 'set1'
    params.process(create_directories=False, create_parameter_hints=False)

    nose.tools.eq_(params.get_path('learner.parameters.value1'), 'learner2')

    params = ParameterContainer().load(filename=os.path.join('material', 'test.parameters.yaml'))
    params['active_set'] = 'set2'
    params.process(create_directories=False, create_parameter_hints=False)

    nose.tools.eq_(params.get_path('learner.parameters.value1'), 'learner3')


def test_override():
    # Test #1
    params = ParameterContainer({
        'field1': 1,
        'field2': 2,
        'field3': 3,
        'field4': 4,
        'subdict': {
            'field1': [1, 2, 3, 4],
            'field2': 100,
        }
    })
    params.override({
        'field1': 11,
        'field3': 13,
        'subdict': {
            'field1': [2, 4],
            'field2': 300,
        }
    })
    nose.tools.eq_(params['field1'], 11)
    nose.tools.eq_(params['field2'], 2)
    nose.tools.eq_(params['field3'], 13)
    nose.tools.eq_(params['field4'], 4)
    nose.tools.eq_(params['subdict']['field1'], [2, 4])
    nose.tools.eq_(params['subdict']['field2'], 300)

    # Test #2
    params = ParameterContainer({
        'field1': 1,
        'field2': 2,
        'field3': 3,
        'field4': 4,
        'subdict': {
            'field1': [1, 2, 3, 4],
            'field2': 100,
        }
    })
    params.override(json.dumps({
            'field1': 11,
            'field3': 13,
            'subdict': {
                'field1': [2, 4],
                'field2': 300,
            }
        })
    )

    nose.tools.eq_(params['field1'], 11)
    nose.tools.eq_(params['field2'], 2)
    nose.tools.eq_(params['field3'], 13)
    nose.tools.eq_(params['field4'], 4)
    nose.tools.eq_(params['subdict']['field1'], [2, 4])
    nose.tools.eq_(params['subdict']['field2'], 300)

    # Test #3
    params = ParameterContainer({
        'field1': 1,
        'field2': 2,
        'field3': 3,
        'field4': 4,
        'subdict': {
            'field1': [1, 2, 3, 4],
            'field2': 100,
        }
    })
    tmp = tempfile.NamedTemporaryFile('r+', suffix='.yaml', dir='/tmp')
    try:
        tmp.write('field1: 10\n')
        tmp.write('field2: 20\n')
        tmp.write('field3: 30\n')
        tmp.write('field4: 40\n')
        tmp.seek(0)

        params.override(tmp.name)

        nose.tools.eq_(params['field1'], 10)
        nose.tools.eq_(params['field2'], 20)
        nose.tools.eq_(params['field3'], 30)
        nose.tools.eq_(params['field4'], 40)
    finally:
        tmp.close()


def test_recipe_parse():
    # Test #1
    test_recipe = 'mel'
    params = ParameterContainer()
    parsed_recipe = params._parse_recipe(recipe=test_recipe)

    # correct amount of items
    nose.tools.eq_(len(parsed_recipe), 1)

    # method is correct
    nose.tools.eq_(parsed_recipe[0]['method'], 'mel')

    # Test #2
    test_recipe = 'mel=0;mfcc=1'
    parsed_recipe = params._parse_recipe(recipe=test_recipe)

    # correct amount of items
    nose.tools.eq_(len(parsed_recipe), 2)

    # methods are correct
    nose.tools.eq_(parsed_recipe[0]['method'], 'mel')
    nose.tools.eq_(parsed_recipe[1]['method'], 'mfcc')

    # vector-index is correct / channel
    nose.tools.eq_(parsed_recipe[0]['vector-index']['channel'], 0)
    nose.tools.eq_(parsed_recipe[1]['vector-index']['channel'], 1)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['full'], True)
    nose.tools.eq_(parsed_recipe[1]['vector-index']['full'], True)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['selection'], False)
    nose.tools.eq_(parsed_recipe[1]['vector-index']['selection'], False)

    # Test #3
    test_recipe = 'mel=1-20'
    parsed_recipe = params._parse_recipe(recipe=test_recipe)

    # correct amount of items
    nose.tools.eq_(len(parsed_recipe), 1)

    # method is correct
    nose.tools.eq_(parsed_recipe[0]['method'], 'mel')

    # vector-index is correct / channel
    nose.tools.eq_(parsed_recipe[0]['vector-index']['channel'], 0)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['full'], False)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['selection'], False)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['start'], 1)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['end'], 21)

    # Test #4
    test_recipe = 'mel=1,2,4,5'
    parsed_recipe = params._parse_recipe(recipe=test_recipe)

    # correct amount of items
    nose.tools.eq_(len(parsed_recipe), 1)

    # extractor is correct
    nose.tools.eq_(parsed_recipe[0]['method'], 'mel')

    # vector-index is correct / channel
    nose.tools.eq_(parsed_recipe[0]['vector-index']['channel'], 0)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['full'], False)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['selection'], True)
    nose.tools.assert_list_equal(parsed_recipe[0]['vector-index']['vector'], [1, 2, 4, 5])

    # Test #5
    test_recipe = 'mel=1:1-20'
    parsed_recipe = params._parse_recipe(recipe=test_recipe)

    # correct amount of items
    nose.tools.eq_(len(parsed_recipe), 1)

    # method is correct
    nose.tools.eq_(parsed_recipe[0]['method'], 'mel')

    # vector-index is correct / channel
    nose.tools.eq_(parsed_recipe[0]['vector-index']['channel'], 1)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['full'], False)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['selection'], False)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['start'], 1)
    nose.tools.eq_(parsed_recipe[0]['vector-index']['end'], 21)
