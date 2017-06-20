#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2017::Acoustic Scene Classification / Baseline System

from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import argparse
import textwrap
import platform

from dcase_framework.application_core import AcousticSceneClassificationAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.learners import SceneClassifier
from dcase_framework.features import FeatureExtractor
from dcase_framework.datasets import AcousticSceneDataset
from dcase_framework.metadata import MetaDataContainer, MetaDataItem

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


class DCASE2013_Scene_EvaluationSet(AcousticSceneDataset):
    """DCASE 2013 Acoustic scene classification, evaluation dataset

    """
    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'DCASE2013-scene-evaluation')
        super(DCASE2013_Scene_EvaluationSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'acoustic scene'
        self.dataset_meta = {
            'authors': 'Dimitrios Giannoulis, Emmanouil Benetos, Dan Stowell, and Mark Plumbley',
            'name_remote': 'IEEE AASP 2013 CASA Challenge - Private Dataset for Scene Classification Task',
            'url': 'http://www.elec.qmul.ac.uk/digitalmusic/sceneseventschallenge/',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': None,
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 5

        self.package_list = [
            {
                'remote_package': 'https://archive.org/download/dcase2013_scene_classification_testset/scenes_stereo_testset.zip',
                'local_package': os.path.join(self.local_path, 'scenes_stereo_testset.zip'),
                'local_audio_path': os.path.join(self.local_path),
            }
        ]

    def _after_extract(self, to_return=None):
        if not self.meta_container.exists():
            meta_data = MetaDataContainer()
            for file in self.audio_files:
                meta_data.append(MetaDataItem({
                    'file': os.path.split(file)[1],
                    'scene_label': os.path.splitext(os.path.split(file)[1])[0][:-2]
                }))
            self.meta_container.update(meta_data)
            self.meta_container.save()

        all_folds_found = True
        for fold in range(1, self.crossvalidation_folds):
            if not os.path.isfile(self._get_evaluation_setup_filename(setup_part='train', fold=fold)):
                all_folds_found = False
            if not os.path.isfile(self._get_evaluation_setup_filename(setup_part='test', fold=fold)):
                all_folds_found = False

        if not all_folds_found:
            if not os.path.isdir(self.evaluation_setup_path):
                os.makedirs(self.evaluation_setup_path)

            classes = self.meta.slice_field('scene_label')
            files = numpy.array(self.meta.slice_field('file'))

            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=self.crossvalidation_folds, test_size=0.3, random_state=0)

            fold = 1
            for train_index, test_index in sss.split(y=classes, X=classes):
                MetaDataContainer(self.meta.filter(file_list=list(files[train_index])),
                                  filename=self._get_evaluation_setup_filename(setup_part='train', fold=fold)).save()

                MetaDataContainer(self.meta.filter(file_list=list(files[test_index])).remove_field('scene_label'),
                                  filename=self._get_evaluation_setup_filename(setup_part='test', fold=fold)).save()

                MetaDataContainer(self.meta.filter(file_list=list(files[test_index])),
                                  filename=self._get_evaluation_setup_filename(setup_part='evaluate', fold=fold)).save()
                fold += 1


class CustomFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        kwargs['valid_extractors'] = [
            'zero_crossing_rate',
            'rmse',
            'centroid',
        ]
        kwargs['default_parameters'] = {
            'zero_crossing_rate': {
                'mono': True,                       # [true, false]
                'center': True,
            },
            'rmse': {
                'mono': True,                       # [true, false]
                'window': 'hamming_asymmetric',     # [hann_asymmetric, hamming_asymmetric]
                'n_fft': 2048,                      # FFT length
            },
            'centroid': {
                'mono': True,                       # [true, false]
                'window': 'hamming_asymmetric',     # [hann_asymmetric, hamming_asymmetric]
                'spectrogram_type': 'magnitude',    # [magnitude, power]
                'n_fft': 2048,                      # FFT length
            },
        }

        super(CustomFeatureExtractor, self).__init__(*args, **kwargs)

    def _zero_crossing_rate(self, data, params):
        """Zero crossing rate

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------

        """

        import librosa

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data[channel, :],
                                                                    frame_length=params.get('win_length_samples'),
                                                                    hop_length=params.get('hop_length_samples'),
                                                                    center=params.get('center')
                                                                    )

            zero_crossing_rate = zero_crossing_rate.reshape((-1, 1))
            feature_matrix.append(zero_crossing_rate)

        return feature_matrix

    def _rmse(self, data, params):
        """Root-mean-square energy

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------

        """
        import librosa

        window = self._window_function(N=params.get('win_length_samples'),
                                       window_type=params.get('window'))

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            S = librosa.magphase(librosa.stft(y=data[channel, :] + self.eps,
                                              n_fft=params.get('n_fft'),
                                              win_length=params.get('win_length_samples'),
                                              hop_length=params.get('hop_length_samples'),
                                              center=False,
                                              window=window)
                                 )[0]
            rmse = librosa.feature.rmse(S=S,
                                        frame_length=params.get('win_length_samples'),
                                        hop_length=params.get('hop_length_samples')
                                        )
            rmse = rmse.reshape((-1, 1))
            feature_matrix.append(rmse)

        return feature_matrix

    def _centroid(self, data, params):
        """Centroid

        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters

        Returns
        -------
        list of numpy.ndarrays
            List of feature matrices, feature matrix per audio channel
        """
        import librosa

        window = self._window_function(N=params.get('win_length_samples'),
                                       window_type=params.get('window'))
        freq = librosa.core.time_frequency.fft_frequencies(sr=params.get('fs'),
                                                           n_fft=params.get('n_fft'))
        freq = freq.reshape((-1, 1))

        feature_matrix = []
        for channel in range(0, data.shape[0]):
            spectrogram_ = self._spectrogram(y=data[channel, :],
                                             n_fft=params.get('n_fft'),
                                             win_length_samples=params.get('win_length_samples'),
                                             hop_length_samples=params.get('hop_length_samples'),
                                             spectrogram_type=params.get('spectrogram_type') if 'spectrogram_type' in params else 'magnitude',
                                             center=True,
                                             window=window)

            centroid = numpy.sum(freq * librosa.util.normalize(spectrogram_, norm=1, axis=0), axis=0, keepdims=True)
            centroid = centroid.reshape((-1, 1))
            feature_matrix.append(centroid)

        return feature_matrix


class SceneClassifierSVM(SceneClassifier):
    """Scene classifier with SVM"""
    def __init__(self, *args, **kwargs):
        super(SceneClassifierSVM, self).__init__(*args, **kwargs)
        self.method = 'svm'

    def learn(self, data, annotations, data_filenames=None):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        from sklearn.svm import SVC

        training_files = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])
        y = numpy.argmax(Y_training, axis=1)

        self['model'] = SVC(**self.learner_params).fit(X_training, y)

        return self

    def _frame_probabilities(self, feature_data):
        if hasattr(self['model'], 'predict_log_proba'):
            return self['model'].predict_log_proba(feature_data).T
        elif hasattr(self['model'], 'predict_proba'):
            return self['model'].predict_proba(feature_data).T
        else:
            message = '{name}: Train model with probability flag [True].'.format(
                name=self.__class__.__name__
            )
            self.logger.exception(message)
            raise AssertionError(message)


class CustomAppCore(AcousticSceneClassificationAppCore):
    def __init__(self, *args, **kwargs):
        kwargs['Datasets'] = {
            'DCASE2013_Scene_EvaluationSet': DCASE2013_Scene_EvaluationSet,
        }
        kwargs['Learners'] = {
            'svm': SceneClassifierSVM,
        }
        kwargs['FeatureExtractor'] = CustomFeatureExtractor

        super(CustomAppCore, self).__init__(*args, **kwargs)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Task 1: Acoustic Scene Classification
            Example how to customize application
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                A system for acoustic scene classification, using DCASE 2013 Challenge evalution dataset.
                Features: mean and std of centroid + zero crossing rate inside 1 second non-overlapping segments
                Classifier: SVM

        '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               os.path.splitext(os.path.basename(__file__))[0]+'.defaults.yaml')
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True
            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

        app = CustomAppCore(name='DCASE 2017::Acoustic Scene Classification / Baseline System',
                            params=params,
                            system_desc=params.get('description'),
                            system_parameter_set_id=params.get('active_set'),
                            setup_label='Development setup',
                            log_system_progress=params.get_path('general.log_system_progress'),
                            show_progress_in_console=params.get_path('general.print_system_progress'),
                            use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                            )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Show evaluated systems
        if args.show_eval:
            app.show_eval()
            return

        # Initialize application
        # ==================================================
        if params['flow']['initialize']:
            app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        if params['flow']['extract_features']:
            app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        if params['flow']['feature_normalizer']:
            app.feature_normalization()

        # System training
        # ==================================================
        if params['flow']['train_system']:
            app.system_training()

        # System evaluation
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']:
                app.system_evaluation()

        # System evaluation in challenge mode
        elif args.mode == 'challenge':
            # Set dataset to testing set for challenge
            params['dataset']['method'] = 'challenge_test'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters('dataset')

            if params['general']['challenge_submission_mode']:
                # If in submission mode, save results in separate folder for easier access
                params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

            challenge_app = CustomAppCore(name='DCASE 2017::Acoustic Scene Classification / Baseline System',
                                          params=params,
                                          system_desc=params.get('description'),
                                          system_parameter_set_id=params.get('active_set'),
                                          setup_label='Evaluation setup',
                                          log_system_progress=params.get_path('general.log_system_progress'),
                                          show_progress_in_console=params.get_path('general.print_system_progress'),
                                          use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                                          )
            # Initialize application
            if params['flow']['initialize']:
                challenge_app.initialize()

            # Extract features for all audio files in the dataset
            if params['flow']['extract_features']:
                challenge_app.feature_extraction()

            # System testing
            if params['flow']['test_system']:
                if params['general']['challenge_submission_mode']:
                    params['general']['overwrite'] = True

                challenge_app.system_testing()

                if params['general']['challenge_submission_mode']:
                    challenge_app.ui.line(" ")
                    challenge_app.ui.line("Results for the challenge are stored at ["+params.get_path('path.recognizer_challenge_output')+"]")
                    challenge_app.ui.line(" ")

            # System evaluation if not in challenge submission mode
            if params['flow']['evaluate_system']:
                challenge_app.system_evaluation()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
