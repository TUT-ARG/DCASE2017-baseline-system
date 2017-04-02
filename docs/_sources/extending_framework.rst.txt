.. _extending_framework:

Extending the framework
=======================

The Framework is designed to be expandable while doing system development with it. The hierarchical class structure enables
easy class extensions, and application cores can be injected with extended classes.

To see working example how to extend the framework, check ``examples/custom.py``.
Below detailed instructions how to extend framework with new datasets, features, and learners.

Adding datasets
---------------

Example how to add the DCASE 2013 Acoustic scene classification evaluation dataset:

.. code-block:: python
    :linenos:

    from dcase_framework.datasets import AcousticSceneDataset
    from dcase_framework.metadata import MetaDataContainer, MetaDataItem

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

Important things to remember:

- Inherit class from the correct base class: ``AcousticSceneDataset`` class for scene classification datasets, ``SoundEventDataset`` class for sound event datasets, and ``AudioTaggingDataset`` for audio tagging datasets.
- Make sure ``meta.txt`` file is generated. This file contains all needed meta data of the dataset.
- Make sure evaluation setup is provided with the dataset or generate your own. Make sure ``train`` and ``test`` methods will return correct data.

Adding features
---------------

Example how to extend FeatureExtractor class with zero crossing rate feature:

.. code-block:: python
    :linenos:

    from dcase_framework.features import FeatureExtractor

    class CustomFeatureExtractor(FeatureExtractor):
        def __init__(self, *args, **kwargs):
            kwargs['valid_extractors'] = [
                'zero_crossing_rate',
            ]
            kwargs['default_parameters'] = {
                'zero_crossing_rate': {
                    'mono': True,
                    'center': True,
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

Important things to remember:

- Decide extractor name
- Add your extractor name to ``valid_extractors`` list
- Add default parameters to ``default_parameters`` dict
- Add extractor method to the class. Method name is extractor name started with underscore.

Addinng learners
----------------

Example how to extend SceneClassifier class with SVM learner:

.. code-block:: python
    :linenos:

    from dcase_framework.learners import SceneClassifier
    from sklearn.svm import SVC

    class SceneClassifierSVM(SceneClassifier):
        """Scene classifier with SVM"""
        def __init__(self, *args, **kwargs):
            super(SceneClassifierSVM, self).__init__(*args, **kwargs)
            self.method = 'svm'

        def learn(self, data, annotations):
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

Important things to remember:

- Inherit class from the correct base class: use ``SceneClassifier`` class for scene classification tasks, ``EventDetector`` class for sound event tasks.
- Implement ``learn`` method for training
- Implement ``predict`` method for testing, or specialize methods from base class to get frame probabilities like in the example.

Extending ApplicationCore
-------------------------

Example how to extend AcousticSceneClassificationAppCore class with all above extensions:

.. code-block:: python
    :linenos:

    from dcase_framework.application_core import AcousticSceneClassificationAppCore

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

