""" Unit tests for SceneClassifierGMM """

import nose.tools
import sys
sys.path.append('..')
import json
import os
from dcase_framework.features import FeatureContainer, FeatureExtractor
from dcase_framework.metadata import MetaDataItem
from dcase_framework.learners import SceneClassifierMLP
import tempfile

learner_params = {
    'seed': 1234,
    'keras': {
        'backend': 'theano',
        'backend_parameters': {
            'floatX': 'float32',
            'device': 'cpu',
        }
    },
    'validation': {
        'enable': False,
    },
    'training': {
        'epochs': 10,
        'batch_size': 16,
        'shuffle': True,
    },
    'model': {
        'config': [
            {
                'class_name': 'Dense',
                'config': {
                    'units': 50,
                    'kernel_initializer': 'uniform',
                    'activation': 'relu',
                }
            },
            {
                'class_name': 'Dense',
                'config': {
                    'units': 'CLASS_COUNT',
                    'kernel_initializer': 'uniform',
                    'activation': 'softmax',
                }
            },
        ],
        'optimizer': {
          'type': 'Adam',
        },
        'loss': 'categorical_crossentropy',
        'metrics': [
            'categorical_accuracy'
        ]
    }
}


def test_learn():
    FeatureExtractor(store=True, overwrite=True).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name='mfcc',
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        },
        storage_paths={
            'mfcc': os.path.join('material', 'test.mfcc.cpickle')
        }
    )

    feature_container = FeatureContainer(filename=os.path.join('material', 'test.mfcc.cpickle'))

    data = {
        'file1.wav': feature_container,
        'file2.wav': feature_container,
    }

    annotations = {
        'file1.wav': MetaDataItem(
            {
                'file': 'file1.wav',
                'scene_label': 'scene1',
            }
        ),
        'file2.wav': MetaDataItem(
            {
                'file': 'file2.wav',
                'scene_label': 'scene2',
            }
        ),
    }

    sc = SceneClassifierMLP(
        method='mlp',
        class_labels=['scene1', 'scene2'],
        params=learner_params,
        filename=os.path.join('material', 'test.model.cpickle'),
        disable_progress_bar=True,
    )

    sc.learn(data=data, annotations=annotations)

    # Test epochs
    nose.tools.eq_(len(sc['learning_history']['loss']), learner_params['training']['epochs'])


def test_predict():
    FeatureExtractor(store=True, overwrite=True).extract(
        audio_file=os.path.join('material', 'test.wav'),
        extractor_name='mfcc',
        extractor_params={
            'mfcc': {
                'n_mfcc': 10
            }
        },
        storage_paths={
            'mfcc': os.path.join('material', 'test.mfcc.cpickle')
        }
    )

    feature_container = FeatureContainer(filename=os.path.join('material', 'test.mfcc.cpickle'))

    data = {
        'file1.wav': feature_container,
        'file2.wav': feature_container,
    }

    annotations = {
        'file1.wav': MetaDataItem(
            {
                'file': 'file1.wav',
                'scene_label': 'scene1',
            }
        ),
        'file2.wav': MetaDataItem(
            {
                'file': 'file2.wav',
                'scene_label': 'scene2',
            }
        ),
    }

    sc = SceneClassifierMLP(
        method='mlp',
        class_labels=['scene1', 'scene2'],
        params=learner_params,
        filename=os.path.join('material', 'test.model.cpickle'),
        disable_progress_bar=True,
    )

    sc.learn(data=data, annotations=annotations)
    recognizer_params = {
        'frame_accumulation': {
            'enable': False,
        },
        'frame_binarization': {
            'enable': True,
            'type': 'frame_max',
        },
        'decision_making': {
            'enable': True,
            'type': 'majority_vote',
        }
    }
    result = sc.predict(
        feature_data=feature_container,
        recognizer_params=recognizer_params
    )

    # Test result
    nose.tools.eq_(len(result) > 0, True)

    # Test errors
    recognizer_params['frame_binarization']['type'] = 'test'
    nose.tools.assert_raises(AssertionError, sc.predict, feature_container, recognizer_params)

    recognizer_params['frame_binarization']['type'] = 'frame_max'
    recognizer_params['decision_making']['type'] = 'test'
    nose.tools.assert_raises(AssertionError, sc.predict, feature_container, recognizer_params)
