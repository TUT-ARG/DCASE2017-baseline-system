#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Datasets
==================
Classes for dataset handling

Dataset - Base class
^^^^^^^^^^^^^^^^^^^^

This is the base class, and all the specialized datasets are inherited from it. One should never use base class itself.

Usage examples:

.. code-block:: python
    :linenos:

    # Create class
    dataset = TUTAcousticScenes_2017_DevelopmentSet(data_path='data')
    # Initialize dataset, this will make sure dataset is downloaded, packages are extracted, and needed meta files are created
    dataset.initialize()
    # Show meta data
    dataset.meta.show()
    # Get all evaluation setup folds
    folds = dataset.folds()
    # Get all evaluation setup folds
    train_data_fold1 = dataset.train(fold=folds[0])
    test_data_fold1 = dataset.test(fold=folds[0])

.. autosummary::
    :toctree: generated/

    Dataset
    Dataset.initialize
    Dataset.show_info
    Dataset.audio_files
    Dataset.audio_file_count
    Dataset.meta
    Dataset.meta_count
    Dataset.error_meta
    Dataset.error_meta_count
    Dataset.fold_count
    Dataset.scene_labels
    Dataset.scene_label_count
    Dataset.event_labels
    Dataset.event_label_count
    Dataset.audio_tags
    Dataset.audio_tag_count
    Dataset.download_packages
    Dataset.extract
    Dataset.train
    Dataset.test
    Dataset.eval
    Dataset.train_files
    Dataset.test_files
    Dataset.eval_files
    Dataset.validation_files
    Dataset.folds
    Dataset.file_meta
    Dataset.file_error_meta
    Dataset.file_error_meta
    Dataset.relative_to_absolute_path
    Dataset.absolute_to_relative

AcousticSceneDataset
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    AcousticSceneDataset

Specialized classes inherited AcousticSceneDataset:

.. autosummary::
    :toctree: generated/

    TUTAcousticScenes_2017_DevelopmentSet
    TUTAcousticScenes_2016_DevelopmentSet
    TUTAcousticScenes_2016_EvaluationSet

SoundEventDataset
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    SoundEventDataset
    SoundEventDataset.event_label_count
    SoundEventDataset.event_labels
    SoundEventDataset.train
    SoundEventDataset.test

Specialized classes inherited SoundEventDataset:

.. autosummary::
    :toctree: generated/

    TUTRareSoundEvents_2017_DevelopmentSet
    TUTSoundEvents_2017_DevelopmentSet
    TUTSoundEvents_2016_DevelopmentSet
    TUTSoundEvents_2016_EvaluationSet

AudioTaggingDataset
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    AudioTaggingDataset


"""

from __future__ import print_function, absolute_import

import sys
import os
import logging
import socket
import zipfile
import tarfile
import collections
import csv
import numpy
import hashlib
import yaml
import shutil
from tqdm import tqdm
from six import iteritems

from .utils import get_parameter_hash, get_class_inheritors
from .decorators import before_and_after_function_wrapper
from .files import TextFile, ParameterFile, ParameterListFile, AudioFile, DictFile
from .containers import DottedDict
from .metadata import MetaDataContainer, MetaDataItem


def dataset_list(data_path, group=None):
    """List of datasets available

    Parameters
    ----------
    data_path : str
        Base path for the datasets
    group : str
        Group label for the datasets, currently supported ['acoustic scene', 'sound event', 'audio tagging']

    Returns
    -------
    str
        Multi line string containing dataset table

    """

    output = ''
    output += '  Dataset list\n'
    output += '  {class_name:<45s} | {group:20s} | {valid:5s} | {files:10s} |\n'.format(
        class_name='Class Name',
        group='Group',
        valid='Valid',
        files='Files'
    )
    output += '  {class_name:<45s} + {group:20s} + {valid:5s} + {files:10s} +\n'.format(
        class_name='-' * 45,
        group='-' * 20,
        valid='-'*5,
        files='-'*10
    )

    def get_empty_row():
        return '  {class_name:<45s} | {group:20s} | {valid:5s} | {files:10s} |\n'.format(
            class_name='',
            group='',
            valid='',
            files=''
        )

    def get_row(d):
        file_count = 0
        if d.meta_container.exists():
            file_count = len(d.meta)

        return '  {class_name:<45s} | {group:20s} | {valid:5s} | {files:10s} |\n'.format(
            class_name=d.__class__.__name__,
            group=d.dataset_group,
            valid='Yes' if d.check_filelist() else 'No',
            files=str(file_count) if file_count else ''
        )

    if not group or group == 'acoustic scene':
        for dataset_class in get_class_inheritors(AcousticSceneDataset):
            d = dataset_class(data_path=data_path)
            output += get_row(d)

    if not group or group == 'sound event':
        for dataset_class in get_class_inheritors(SoundEventDataset):
            d = dataset_class(data_path=data_path)
            output += get_row(d)

    if not group or group == 'audio tagging':
        for dataset_class in get_class_inheritors(AudioTaggingDataset):
            d = dataset_class(data_path=data_path)
            output += get_row(d)

    return output


def dataset_factory(*args, **kwargs):
    """Factory to get correct dataset class based on name

    Parameters
    ----------
    dataset_class_name : str
        Class name
        Default value "None"

    Raises
    ------
    NameError
        Class does not exists

    Returns
    -------
    Dataset class

    """

    dataset_class_name = kwargs.get('dataset_class_name', None)
    try:
        return eval(dataset_class_name)(*args, **kwargs)
    except NameError:
        message = '{name}: No valid dataset given [{dataset_class_name}]'.format(
            name='dataset_factory',
            dataset_class_name=dataset_class_name
        )

        logging.getLogger('dataset_factory').exception(message)
        raise NameError(message)


class Dataset(object):
    """Dataset base class

    The specific dataset classes are inherited from this class, and only needed methods are reimplemented.

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        name : str
        storage_name : str
        data_path : str
            Basepath where the dataset is stored.
            (Default value='data')
        logger : logger
            Instance of logging
            Default value "none"
        show_progress_in_console : bool
            Show progress in console.
            Default value "True"
        log_system_progress : bool
            Show progress in log.
            Default value "False"
        use_ascii_progress_bar : bool
            Show progress bar using ASCII characters. Use this if your console does not support UTF-8 characters.
            Default value "False"
        """

        self.logger = kwargs.get('logger') or logging.getLogger(__name__)

        self.disable_progress_bar = not kwargs.get('show_progress_in_console', True)
        self.log_system_progress = kwargs.get('log_system_progress', False)
        self.use_ascii_progress_bar = kwargs.get('use_ascii_progress_bar', True)

        # Dataset name
        self.name = kwargs.get('name', 'dataset')

        # Folder name for dataset
        self.storage_name = kwargs.get('storage_name', 'dataset')

        # Path to the dataset
        self.local_path = os.path.join(kwargs.get('data_path', 'data'), self.storage_name)

        # Evaluation setup folder
        self.evaluation_setup_folder = kwargs.get('evaluation_setup_folder', 'evaluation_setup')

        # Path to the folder containing evaluation setup files
        self.evaluation_setup_path = os.path.join(self.local_path, self.evaluation_setup_folder)

        # Meta data file, csv-format
        self.meta_filename = kwargs.get('meta_filename', 'meta.txt')

        # Path to meta data file
        self.meta_container = MetaDataContainer(filename=os.path.join(self.local_path, self.meta_filename))
        if self.meta_container.exists():
            self.meta_container.load()

        # Error meta data file, csv-format
        self.error_meta_filename = kwargs.get('error_meta_filename', 'error.txt')

        # Path to error meta data file
        self.error_meta_file = os.path.join(self.local_path, self.error_meta_filename)

        # Hash file to detect removed or added files
        self.filelisthash_filename = kwargs.get('filelisthash_filename', 'filelist.python.hash')

        # Dirs to be excluded when calculating filelist hash
        self.filelisthash_exclude_dirs = kwargs.get('filelisthash_exclude_dirs', [])

        # Number of evaluation folds
        self.crossvalidation_folds = 1

        # List containing dataset package items
        # Define this in the inherited class.
        # Format:
        # {
        #        'remote_package': download_url,
        #        'local_package': os.path.join(self.local_path, 'name_of_downloaded_package'),
        #        'local_audio_path': os.path.join(self.local_path, 'name_of_folder_containing_audio_files'),
        # }
        self.package_list = []

        # List of audio files
        self.files = None

        # List of audio error meta data dict
        self.error_meta_data = None

        # Training meta data for folds
        self.crossvalidation_data_train = {}

        # Testing meta data for folds
        self.crossvalidation_data_test = {}

        # Evaluation meta data for folds
        self.crossvalidation_data_eval = {}

        # Recognized audio extensions
        self.audio_extensions = {'wav', 'flac'}

        self.default_audio_extension = 'wav'

        # Reference data presence flag, by default dataset should have reference data present.
        # However, some evaluation dataset might not have
        self.reference_data_present = True

        # Info fields for dataset
        self.authors = ''
        self.name_remote = ''
        self.url = ''
        self.audio_source = ''
        self.audio_type = ''
        self.recording_device_model = ''
        self.microphone_model = ''

    def initialize(self):
        # Create the dataset path if does not exist
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)

        if not self.check_filelist():
            self.download_packages()
            self.extract()
            self._save_filelist_hash()

        return self

    def show_info(self):
        DottedDict(self.dataset_meta).show()

    @property
    def audio_files(self):
        """Get all audio files in the dataset

        Parameters
        ----------

        Returns
        -------
        filelist : list
            File list with absolute paths

        """

        if self.files is None:
            self.files = []
            for item in self.package_list:
                path = item['local_audio_path']
                if path:
                    l = os.listdir(path)
                    for f in l:
                        file_name, file_extension = os.path.splitext(f)
                        if file_extension[1:] in self.audio_extensions:
                            if os.path.abspath(os.path.join(path, f)) not in self.files:
                                self.files.append(os.path.abspath(os.path.join(path, f)))
            self.files.sort()
        return self.files

    @property
    def audio_file_count(self):
        """Get number of audio files in dataset

        Parameters
        ----------

        Returns
        -------
        filecount : int
            Number of audio files

        """

        return len(self.audio_files)

    @property
    def meta(self):
        """Get meta data for dataset. If not already read from disk, data is read and returned.

        Parameters
        ----------

        Returns
        -------
        meta_container : list
            List containing meta data as dict.

        Raises
        -------
        IOError
            meta file not found.

        """

        if self.meta_container.empty():
            if self.meta_container.exists():
                self.meta_container.load()
            else:
                message = '{name}: Meta file not found [{filename}]'.format(
                    name=self.__class__.__name__,
                    filename=self.meta_container.filename
                )

                self.logger.exception(message)
                raise IOError(message)

        return self.meta_container

    @property
    def meta_count(self):
        """Number of meta data items.

        Parameters
        ----------

        Returns
        -------
        meta_item_count : int
            Meta data item count

        """

        return len(self.meta_container)

    @property
    def error_meta(self):
        """Get audio error meta data for dataset. If not already read from disk, data is read and returned.

        Parameters
        ----------

        Raises
        -------
        IOError:
            audio error meta file not found.

        Returns
        -------
        error_meta_data : list
            List containing audio error meta data as dict.

        """

        if self.error_meta_data is None:
            self.error_meta_data = MetaDataContainer(filename=self.error_meta_file)
            if self.error_meta_data.exists():
                self.error_meta_data.load()
            else:
                message = '{name}: Error meta file not found [{filename}]'.format(name=self.__class__.__name__,
                                                                                  filename=self.error_meta_file)
                self.logger.exception(message)
                raise IOError(message)

        return self.error_meta_data

    def error_meta_count(self):
        """Number of error meta data items.

        Parameters
        ----------

        Returns
        -------
        meta_item_count : int
            Meta data item count

        """

        return len(self.error_meta)

    @property
    def fold_count(self):
        """Number of fold in the evaluation setup.

        Parameters
        ----------

        Returns
        -------
        fold_count : int
            Number of folds

        """

        return self.crossvalidation_folds

    @property
    def scene_labels(self):
        """List of unique scene labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        labels : list
            List of scene labels in alphabetical order.

        """

        return self.meta_container.unique_scene_labels

    @property
    def scene_label_count(self):
        """Number of unique scene labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        scene_label_count : int
            Number of unique scene labels.

        """

        return self.meta_container.scene_label_count

    def event_labels(self):
        """List of unique event labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        labels : list
            List of event labels in alphabetical order.

        """

        return self.meta_container.unique_event_labels

    @property
    def event_label_count(self):
        """Number of unique event labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        event_label_count : int
            Number of unique event labels

        """

        return self.meta_container.event_label_count

    @property
    def audio_tags(self):
        """List of unique audio tags in the meta data.

        Parameters
        ----------

        Returns
        -------
        labels : list
            List of audio tags in alphabetical order.

        """

        tags = []
        for item in self.meta:
            if 'tags' in item:
                for tag in item['tags']:
                    if tag and tag not in tags:
                        tags.append(tag)
        tags.sort()
        return tags

    @property
    def audio_tag_count(self):
        """Number of unique audio tags in the meta data.

        Parameters
        ----------

        Returns
        -------
        audio_tag_count : int
            Number of unique audio tags

        """

        return len(self.audio_tags)

    def __getitem__(self, i):
        """Getting meta data item

        Parameters
        ----------
        i : int
            item id

        Returns
        -------
        meta_data : dict
            Meta data item
        """
        if i < len(self.meta_container):
            return self.meta_container[i]
        else:
            return None

    def __iter__(self):
        """Iterator for meta data items

        Parameters
        ----------
        Nothing

        Returns
        -------
        Nothing

        """

        i = 0
        meta = self[i]

        # yield window while it's valid
        while meta is not None:
            yield meta
            # get next item
            i += 1
            meta = self[i]

    def download_packages(self):
        """Download dataset packages over the internet to the local path

        Parameters
        ----------

        Returns
        -------
        Nothing

        Raises
        -------
        IOError
            Download failed.

        """

        try:
            from urllib.request import urlretrieve
        except ImportError:
            from urllib import urlretrieve

        # Set socket timeout
        socket.setdefaulttimeout(120)

        item_progress = tqdm(self.package_list,
                             desc="{0: <25s}".format('Download package list'),
                             file=sys.stdout,
                             leave=False,
                             disable=self.disable_progress_bar,
                             ascii=self.use_ascii_progress_bar)

        for item in item_progress:
            try:
                if item['remote_package'] and not os.path.isfile(item['local_package']):

                    def progress_hook(t):
                        """
                        Wraps tqdm instance. Don't forget to close() or __exit__()
                        the tqdm instance once you're done with it (easiest using `with` syntax).
                        """

                        last_b = [0]

                        def inner(b=1, bsize=1, tsize=None):
                            """
                            b  : int, optional
                                Number of blocks just transferred [default: 1].
                            bsize  : int, optional
                                Size of each block (in tqdm units) [default: 1].
                            tsize  : int, optional
                                Total size (in tqdm units). If [default: None] remains unchanged.
                            """
                            if tsize is not None:
                                t.total = tsize
                            t.update((b - last_b[0]) * bsize)
                            last_b[0] = b

                        return inner

                    remote_file = item['remote_package']
                    tmp_file = os.path.join(self.local_path, 'tmp_file')
                    with tqdm(desc="{0: >25s}".format(os.path.splitext(remote_file.split('/')[-1])[0]),
                              file=sys.stdout,
                              unit='B',
                              unit_scale=True,
                              miniters=1,
                              leave=False,
                              disable=self.disable_progress_bar,
                              ascii=self.use_ascii_progress_bar) as t:

                        local_filename, headers = urlretrieve(
                            remote_file,
                            filename=tmp_file,
                            reporthook=progress_hook(t),
                            data=None
                        )

                    os.rename(tmp_file, item['local_package'])

            except Exception as e:
                message = '{name}: Download failed [{filename}] [{errno}: {strerror}]'.format(
                    name=self.__class__.__name__,
                    filename=item['remote_package'],
                    errno=e.errno if hasattr(e, 'errno') else '',
                    strerror=e.strerror if hasattr(e, 'strerror') else '',
                )
                self.logger.exception(message)
                raise

    @before_and_after_function_wrapper
    def extract(self):
        """Extract the dataset packages

        Parameters
        ----------

        Returns
        -------
        Nothing

        """

        item_progress = tqdm(self.package_list,
                             desc="{0: <25s}".format('Extract packages'),
                             file=sys.stdout,
                             leave=False,
                             disable=self.disable_progress_bar,
                             ascii=self.use_ascii_progress_bar)

        for item_id, item in enumerate(item_progress):
            if item['local_package'] and os.path.isfile(item['local_package']):
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {package:<30s}'.format(
                        title='Extract packages ',
                        item_id=item_id,
                        total=len(item_progress),
                        package=item['local_package'])
                    )

                if item['local_package'].endswith('.zip'):

                    with zipfile.ZipFile(item['local_package'], "r") as z:
                        # Trick to omit first level folder
                        parts = []
                        for name in z.namelist():
                            if not name.endswith('/'):
                                parts.append(name.split('/')[:-1])
                        prefix = os.path.commonprefix(parts) or ''

                        if prefix:
                            if len(prefix) > 1:
                                prefix_ = list()
                                prefix_.append(prefix[0])
                                prefix = prefix_

                            prefix = '/'.join(prefix) + '/'
                        offset = len(prefix)

                        # Start extraction
                        members = z.infolist()
                        file_count = 1
                        progress = tqdm(members,
                                        desc="{0: <25s}".format('Extract'),
                                        file=sys.stdout,
                                        leave=False,
                                        disable=self.disable_progress_bar,
                                        ascii=self.use_ascii_progress_bar)
                        for i, member in enumerate(progress):
                            if self.log_system_progress:
                                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                    title='Extract ',
                                    item_id=i,
                                    total=len(progress),
                                    file=member.filename)
                                )

                            if len(member.filename) > offset:
                                member.filename = member.filename[offset:]
                                progress.set_description("{0: >35s}".format(member.filename.split('/')[-1]))
                                progress.update()
                                if not os.path.isfile(os.path.join(self.local_path, member.filename)):
                                    try:
                                        if 'zip_password' in item:
                                            z.extract(member=member, path=self.local_path, pwd=item['zip_password'])
                                        else:
                                            z.extract(member=member, path=self.local_path)
                                    except KeyboardInterrupt:
                                        # Delete latest file, since most likely it was not extracted fully
                                        os.remove(os.path.join(self.local_path, member.filename))

                                        # Quit
                                        sys.exit()

                                file_count += 1

                elif item['local_package'].endswith('.tar.gz'):
                    tar = tarfile.open(item['local_package'], "r:gz")
                    progress = tqdm(tar,
                                    desc="{0: <25s}".format('Extract'),
                                    file=sys.stdout,
                                    leave=False,
                                    disable=self.disable_progress_bar,
                                    ascii=self.use_ascii_progress_bar)
                    for i, tar_info in enumerate(progress):
                        if self.log_system_progress:
                            self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                title='Extract ',
                                item_id=i,
                                total=len(progress),
                                file=tar_info.name)
                            )

                        if not os.path.isfile(os.path.join(self.local_path, tar_info.name)):
                            tar.extract(tar_info, self.local_path)
                        tar.members = []
                    tar.close()

    def _get_filelist(self, exclude_dirs=None):
        """List of files under local_path

        Parameters
        ----------
        exclude_dirs : list of str
            List of directories to be excluded
            Default value "[]"

        Returns
        -------
        filelist: list
            File list
        """

        if exclude_dirs is None:
            exclude_dirs = []

        filelist = []
        for path, subdirs, files in os.walk(self.local_path):
            for name in files:
                if os.path.splitext(name)[1] != os.path.splitext(self.filelisthash_filename)[1] and os.path.split(path)[1] not in exclude_dirs:
                    filelist.append(os.path.join(path, name))

        return sorted(filelist)

    def check_filelist(self):
        """Generates hash from file list and check does it matches with one saved in filelist.hash.
        If some files have been deleted or added, checking will result False.

        Parameters
        ----------

        Returns
        -------
        result: bool
            Result

        """

        if os.path.isfile(os.path.join(self.local_path, self.filelisthash_filename)):
            old_hash_value = TextFile(filename=os.path.join(self.local_path, self.filelisthash_filename)).load()[0]
            file_list = self._get_filelist(exclude_dirs=self.filelisthash_exclude_dirs)
            new_hash_value = get_parameter_hash(file_list)
            if old_hash_value != new_hash_value:
                return False
            else:
                return True
        else:
            return False

    def _save_filelist_hash(self):
        """Generates file list hash, and saves it as filelist.hash under local_path.

        Parameters
        ----------
        Nothing

        Returns
        -------
        Nothing

        """

        filelist = self._get_filelist()

        hash_value = get_parameter_hash(filelist)

        TextFile([hash_value], filename=os.path.join(self.local_path, self.filelisthash_filename)).save()

    def train(self, fold=0):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = []
            if fold > 0:
                self.crossvalidation_data_train[fold] = MetaDataContainer(
                    filename=self._get_evaluation_setup_filename(setup_part='train', fold=fold)).load()

            else:
                self.crossvalidation_data_train[0] = self.meta_container

            for item in self.crossvalidation_data_train[fold]:
                item['file'] = self.relative_to_absolute_path(item['file'])

        return self.crossvalidation_data_train[fold]

    def test(self, fold=0):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = []
            if fold > 0:
                self.crossvalidation_data_test[fold] = MetaDataContainer(
                    filename=self._get_evaluation_setup_filename(setup_part='test', fold=fold)).load()

                for item in self.crossvalidation_data_test[fold]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

            else:
                self.crossvalidation_data_test[fold] = self.meta_container

            for item in self.crossvalidation_data_test[fold]:
                item['file'] = self.relative_to_absolute_path(item['file'])

        return self.crossvalidation_data_test[fold]

    def eval(self, fold=0):
        """List of evaluation items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_eval:
            self.crossvalidation_data_eval[fold] = []
            if fold > 0:
                self.crossvalidation_data_eval[fold] = MetaDataContainer(
                    filename=self._get_evaluation_setup_filename(setup_part='evaluate', fold=fold)).load()

            else:
                self.crossvalidation_data_eval[fold] = self.meta_container

            for item in self.crossvalidation_data_eval[fold]:
                item['file'] = self.relative_to_absolute_path(item['file'])

        return self.crossvalidation_data_eval[fold]

    def train_files(self, fold=0):
        """List of training files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of filenames
            List containing all files assigned to training set for given fold.

        """

        return self.train(fold=fold).file_list

    def test_files(self, fold=0):
        """List of testing files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of filenames
            List containing all files assigned to testing set for given fold.

        """

        return self.test(fold=fold).file_list

    def eval_files(self, fold=0):
        """List of evaluation files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of filenames
            List containing all files assigned to testing set for given fold.

        """

        return self.eval(fold=fold).file_list

    def validation_files(self, fold=0):
        """List of validation files if they are specified by the dataset. Validation files are always subset of training files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of filenames
            List containing all files assigned to training set for given fold.
            (Default value=[])

        """
        return []

    def folds(self, mode='folds'):
        """List of fold ids

        Parameters
        ----------
        mode : str {'folds','full'}
            Fold setup type, possible values are 'folds' and 'full'. In 'full' mode fold number is set 0 and all data is used for training.
            (Default value=folds)

        Returns
        -------
        list : list of integers
            Fold ids

        """

        if mode == 'folds':
            return range(1, self.crossvalidation_folds + 1)
        elif mode == 'full':
            return [0]

    def file_meta(self, filename):
        """Meta data for given file

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        list : list of dicts
            List containing all meta data related to given file.

        """

        return self.meta_container.filter(filename=self.absolute_to_relative(filename))

    def file_error_meta(self, filename):
        """Error meta data for given file

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        list : list of dicts
            List containing all error meta data related to given file.

        """

        return self.error_meta.filter(filename=self.absolute_to_relative(filename))

    def relative_to_absolute_path(self, path):
        """Converts relative path into absolute path.

        Parameters
        ----------
        path : str
            Relative path

        Returns
        -------
        path : str
            Absolute path

        """

        return os.path.abspath(os.path.expanduser(os.path.join(self.local_path, path)))

    def absolute_to_relative(self, path):
        """Converts absolute path into relative path.

        Parameters
        ----------
        path : str
            Absolute path

        Returns
        -------
        path : str
            Relative path

        """

        if path.startswith(os.path.abspath(self.local_path)):
            return os.path.relpath(path, self.local_path)
        else:
            return path

    def _get_evaluation_setup_filename(self, setup_part='train', fold=None, scene_label=None, file_extension='txt'):
        parts = []
        if scene_label:
            parts.append(scene_label)

        if fold:
            parts.append('fold' + str(fold))

        if setup_part == 'train':
            parts.append('train')
        elif setup_part == 'test':
            parts.append('test')
        elif setup_part == 'evaluate':
            parts.append('evaluate')

        return os.path.join(self.evaluation_setup_path, '_'.join(parts) + '.' + file_extension)


class AcousticSceneDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(AcousticSceneDataset, self).__init__(*args, **kwargs)
        self.dataset_group = 'base class'


class SoundEventDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(SoundEventDataset, self).__init__(*args, **kwargs)
        self.dataset_group = 'base class'

    def event_label_count(self, scene_label=None):
        """Number of unique scene labels in the meta data.

        Parameters
        ----------
        scene_label : str
            Scene label
            Default value "None"

        Returns
        -------
        scene_label_count : int
            Number of unique scene labels.

        """

        return len(self.event_labels(scene_label=scene_label))

    def event_labels(self, scene_label=None):
        """List of unique event labels in the meta data.

        Parameters
        ----------
        scene_label : str
            Scene label
            Default value "None"

        Returns
        -------
        labels : list
            List of event labels in alphabetical order.

        """

        if scene_label is not None:
            labels = self.meta_container.filter(scene_label=scene_label).unique_event_labels
        else:
            labels = self.meta_container.unique_event_labels

        labels.sort()
        return labels

    def train(self, fold=0, scene_label=None, event_label=None):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_train[fold]:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer()

                if fold > 0:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(setup_part='train', fold=fold, scene_label=scene_label_)).load()
                else:
                    self.crossvalidation_data_train[0][scene_label_] = self.meta_container.filter(
                        scene_label=scene_label_
                    )

                for item in self.crossvalidation_data_train[fold][scene_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

        if scene_label:
            return self.crossvalidation_data_train[fold][scene_label]

        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_train[fold][scene_label_]

            return data

    def test(self, fold=0, scene_label=None, event_label=None):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer()
                if fold > 0:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='test', fold=fold, scene_label=scene_label_)
                    ).load()

                else:
                    self.crossvalidation_data_test[0][scene_label_] = self.meta_container.filter(
                        scene_label=scene_label_
                    )

                for item in self.crossvalidation_data_test[fold][scene_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

        if scene_label:
            return self.crossvalidation_data_test[fold][scene_label]

        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_test[fold][scene_label_]

            return data

    def eval(self, fold=0, scene_label=None, event_label=None):
        """List of evaluation items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_eval:
            self.crossvalidation_data_eval[fold] = []
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_eval[fold]:
                    self.crossvalidation_data_eval[fold][scene_label_] = MetaDataContainer()
                if fold > 0:
                    self.crossvalidation_data_eval[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='test', fold=fold, scene_label=scene_label_)
                    ).load()

                else:
                    self.crossvalidation_data_eval[0][scene_label_] = self.meta_container.filter(
                        scene_label=scene_label_
                    )

                for item in self.crossvalidation_data_eval[fold][scene_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

        if scene_label:
            return self.crossvalidation_data_eval[fold][scene_label]

        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_eval[fold][scene_label_]

            return data

    def train_files(self, fold=0, scene_label=None, event_label=None):
        """List of training files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")
        event_label : str
            Event label
            (Default value "None")

        Returns
        -------
        list : list of filenames
            List containing all files assigned to training set for given fold.

        """

        return self.train(fold=fold, scene_label=scene_label, event_label=event_label).file_list

    def test_files(self, fold=0, scene_label=None, event_label=None):
        """List of testing files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")
        event_label : str
            Event label
            (Default value "None")

        Returns
        -------
        list : list of filenames
            List containing all files assigned to testing set for given fold.

        """

        return self.test(fold=fold, scene_label=scene_label, event_label=event_label).file_list

    def eval_files(self, fold=0, scene_label=None, event_label=None):
        """List of evaluation files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")
        event_label : str
            Event label
            (Default value "None")

        Returns
        -------
        list : list of filenames
            List containing all files assigned to testing set for given fold.

        """

        return self.eval(fold=fold, scene_label=scene_label, event_label=event_label).file_list

    def validation_files(self, fold=0, scene_label=None, event_label=None):
        """List of validation files if they are specified by the dataset. Validation files are always subset of training files.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            (Default value "None")
        event_label : str
            Event label
            (Default value "None")

        Returns
        -------
        list : list of filenames
            List containing all files assigned to training set for given fold.
            Default value=[]
        """

        return []


class SyntheticSoundEventDataset(SoundEventDataset):
    def __init__(self, *args, **kwargs):
        super(SyntheticSoundEventDataset, self).__init__(*args, **kwargs)
        self.dataset_group = 'base class'

    def initialize(self):
        # Create the dataset path if does not exist
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)

        if not self.check_filelist():
            self.download_packages()
            self.extract()
            self._save_filelist_hash()

        self.synthesize()

        return self

    @before_and_after_function_wrapper
    def synthesize(self):
        pass


class AudioTaggingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(AudioTaggingDataset, self).__init__(*args, **kwargs)
        self.dataset_group = 'base class'


# =====================================================
# DCASE 2017
# =====================================================
class TUTAcousticScenes_2017_DevelopmentSet(AcousticSceneDataset):
    """TUT Acoustic scenes 2017 development dataset

    This dataset is used in DCASE2017 - Task 1, Acoustic scene classification

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-acoustic-scenes-2017-development')
        super(TUTAcousticScenes_2017_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'acoustic scene'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Acoustic Scenes 2017, development dataset',
            'url': None,
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 4

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.error.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.error.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.6.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.6.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.7.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.7.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.8.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.8.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.9.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.9.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.10.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-development.audio.10.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            meta_data = collections.OrderedDict()
            for fold in range(1, self.crossvalidation_folds):
                # Read train files in
                fold_data = MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt')).load()

                fold_data += MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_evaluate.txt')).load()

                for item in fold_data:
                    if item['file'] not in meta_data:
                        raw_path, raw_filename = os.path.split(item['file'])
                        relative_path = self.absolute_to_relative(raw_path)
                        item['file'] = os.path.join(relative_path, raw_filename)
                        item['identifier'] = raw_filename.split('_')[0]
                        meta_data[item['file']] = item

            self.meta_container.update(meta_data.values())
            self.meta_container.save()
        else:
            self.meta_container.load()

    def train(self, fold=0):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = []
            if fold > 0:
                self.crossvalidation_data_train[fold] = MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt')).load()

                for item in self.crossvalidation_data_train[fold]:
                    item['file'] = self.relative_to_absolute_path(item['file'])
                    raw_path, raw_filename = os.path.split(item['file'])
                    location_id = raw_filename.split('_')[0]
                    item['identifier'] = location_id
            else:
                self.crossvalidation_data_train[0] = self.meta_container

        return self.crossvalidation_data_train[fold]


class TUTAcousticScenes_2017_EvaluationSet(AcousticSceneDataset):
    """TUT Acoustic scenes 2017 evaluation dataset

    This dataset is used in DCASE2017 - Task 1, Acoustic scene classification

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-acoustic-scenes-2017-evaluation')
        super(TUTAcousticScenes_2017_EvaluationSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'acoustic scene'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Acoustic Scenes 2017, development dataset',
            'url': None,
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 1
        source_url = 'https://zenodo.org/record/1040168/files/'
        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'street'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-acoustic-scenes-2017-evaluation.audio.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2017-evaluation.audio.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            if os.path.isfile(os.path.join(self.evaluation_setup_path, 'evaluate.txt')):
                meta_data = collections.OrderedDict()

                # Read files in
                data = MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'evaluate.txt')
                ).load()

                # Load filename mapping
                map_filename = os.path.join(self.evaluation_setup_path, 'map.txt')
                filename_map = {}
                if os.path.exists(map_filename):
                    with open(map_filename, 'rt') as f:
                        for row in csv.reader(f, delimiter='\t'):
                            filename_map[row[1]] = row[0]

                for item in data:
                    if item['file'] not in meta_data:
                        raw_path, raw_filename = os.path.split(item['file'])
                        relative_path = self.absolute_to_relative(raw_path)
                        item['file'] = os.path.join(relative_path, raw_filename)

                        # Recover location identifier from ordinal filename
                        if item['file'] in filename_map:
                            item['identifier'] = os.path.split(filename_map[item['file']])[1].split('_')[0]

                        meta_data[item['file']] = item

                self.meta_container.update(meta_data.values())
                self.meta_container.save()

        else:
            self.meta_container.load()

    def train(self, fold=0):
        return []

    def test(self, fold=0):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.crossvalidation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
            else:
                with open(os.path.join(self.evaluation_setup_path, 'test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.crossvalidation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})

        return self.crossvalidation_data_test[fold]


class TUTRareSoundEvents_2017_DevelopmentSet(SyntheticSoundEventDataset):
    """TUT Acoustic scenes 2017 development dataset

    This dataset is used in DCASE2017 - Task 2, Rare sound event detection

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-rare-sound-events-2017-development')
        kwargs['filelisthash_exclude_dirs'] = kwargs.get('filelisthash_exclude_dirs', ['generated_data'])

        self.dcase_compatibility = kwargs.get('dcase_compatibility', True)

        self.synth_parameters = DottedDict({
            'train': {
                'seed': 42,
                'mixture': {
                    'fs': 44100,
                    'bitdepth': 24,
                    'length_seconds': 30.0,
                    'anticlipping_factor': 0.2,
                },
                'event_presence_prob': 0.5,
                'mixtures_per_class': 500,
                'ebr_list': [-6, 0, 6],
            },
            'test': {
                'seed': 42,
                'mixture': {
                    'fs': 44100,
                    'bitdepth': 24,
                    'length_seconds': 30.0,
                    'anticlipping_factor': 0.2,
                },
                'event_presence_prob': 0.5,
                'mixtures_per_class': 500,
                'ebr_list': [-6, 0, 6],
            }
        })

        # Override synth parameters
        if kwargs.get('synth_parameters'):
            self.synth_parameters.merge(kwargs.get('synth_parameters'))

        # Meta filename depends on synth parameters
        meta_filename = 'meta_'+self.synth_parameters.get_hash_for_path()+'.txt'
        kwargs['meta_filename'] = kwargs.get('meta_filename', os.path.join('generated_data', meta_filename))

        # Initialize baseclass
        super(TUTRareSoundEvents_2017_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Aleksandr Diment, Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Rare Sound Events 2017, development dataset',
            'url': None,
            'audio_source': 'Synthetic',
            'audio_type': 'Natural',
            'recording_device_model': 'Unknown',
            'microphone_model': 'Unknown',
        }

        self.crossvalidation_folds = 1

        source_url = 'https://zenodo.org/record/401395/files/'

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.code.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.code.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.6.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.6.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.7.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.7.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.8.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_bgs_and_cvsetup.8.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url+'TUT-rare-sound-events-2017-development.source_data_events.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-development.source_data_events.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    @property
    def event_labels(self, scene_label=None):
        """List of unique event labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        labels : list
            List of event labels in alphabetical order.

        """

        labels = ['babycry', 'glassbreak', 'gunshot']
        labels.sort()
        return labels

    def train(self, fold=0, scene_label=None, event_label=None):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        event_label : str
            Event label
            Default value "None"

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = {}

            for event_label_ in self.event_labels:
                if event_label_ not in self.crossvalidation_data_train[fold]:
                    self.crossvalidation_data_train[fold][event_label_] = MetaDataContainer()

                if fold == 1:
                    params_hash = self.synth_parameters.get_hash_for_path('train')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtrain_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_devtrain_' + event_label_ + '.csv'
                    )

                    self.crossvalidation_data_train[fold][event_label_] = MetaDataContainer(
                        filename=event_list_filename).load()

                elif fold == 0:
                    params_hash = self.synth_parameters.get_hash_for_path('train')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtrain_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_devtrain_' + event_label_ + '.csv'
                    )

                    # Load train files
                    self.crossvalidation_data_train[0][event_label_] = MetaDataContainer(
                        filename=event_list_filename).load()

                    params_hash = self.synth_parameters.get_hash_for_path('test')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtest_' + params_hash,
                        'meta'
                    )
                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_devtest_' + event_label_ + '.csv'
                    )

                    # Load test files
                    self.crossvalidation_data_train[0][event_label_] += MetaDataContainer(
                        filename=event_list_filename).load()

                for item in self.crossvalidation_data_train[fold][event_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

        if event_label:
            return self.crossvalidation_data_train[fold][event_label]
        else:
            data = MetaDataContainer()
            for event_label_ in self.event_labels:
                data += self.crossvalidation_data_train[fold][event_label_]

            return data

    def test(self, fold=0, scene_label=None, event_label=None):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        event_label : str
            Event label
            Default value "None"

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for event_label_ in self.event_labels:
                if event_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][event_label_] = MetaDataContainer()
                if fold == 1:
                    params_hash = self.synth_parameters.get_hash_for_path('test')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtest_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(mixture_meta_path, 'event_list_devtest_' + event_label_ + '.csv')

                    self.crossvalidation_data_test[fold][event_label_] = MetaDataContainer(
                        filename=event_list_filename
                    ).load()

                elif fold == 0:
                    params_hash = self.synth_parameters.get_hash_for_path('train')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtrain_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_devtrain_' + event_label_ + '.csv'
                    )

                    # Load train files
                    self.crossvalidation_data_test[0][event_label_] = MetaDataContainer(
                        filename=event_list_filename
                    ).load()

                    params_hash = self.synth_parameters.get_hash_for_path('test')
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_devtest_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_devtest_' + event_label_ + '.csv'
                    )

                    # Load test files
                    self.crossvalidation_data_test[0][event_label_] += MetaDataContainer(
                        filename=event_list_filename
                    ).load()

                for item in self.crossvalidation_data_test[fold][event_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])

        if event_label:
            return self.crossvalidation_data_test[fold][event_label]
        else:
            data = MetaDataContainer()
            for event_label_ in self.event_labels:
                data += self.crossvalidation_data_test[fold][event_label_]

            return data

    @before_and_after_function_wrapper
    def synthesize(self):
        subset_map = {'train': 'devtrain',
                      'test': 'devtest'}

        background_audio_path = os.path.join(self.local_path, 'data', 'source_data', 'bgs')
        event_audio_path = os.path.join(self.local_path, 'data', 'source_data', 'events')
        cv_setup_path = os.path.join(self.local_path, 'data', 'source_data', 'cv_setup')

        set_progress = tqdm(['train', 'test'],
                            desc="{0: <25s}".format('Set'),
                            file=sys.stdout,
                            leave=False,
                            disable=self.disable_progress_bar,
                            ascii=self.use_ascii_progress_bar)

        for subset_label in set_progress:
            if self.log_system_progress:
                self.logger.info('  {title:<15s} [{subset_label:<30s}]'.format(
                    title='Set ',
                    subset_label=subset_label)
                )

            subset_name_on_disk = subset_map[subset_label]

            background_meta = ParameterListFile().load(filename=os.path.join(cv_setup_path, 'bgs_' + subset_name_on_disk + '.yaml'))
            event_meta = ParameterFile().load(
                filename=os.path.join(cv_setup_path, 'events_' + subset_name_on_disk + '.yaml')
            )

            params = self.synth_parameters.get_path(subset_label)
            params_hash = self.synth_parameters.get_hash_for_path(subset_label)

            r = numpy.random.RandomState(params.get('seed', 42))

            mixture_path = os.path.join(
                self.local_path,
                'generated_data',
                'mixtures_' + subset_name_on_disk + '_' + params_hash
            )

            mixture_audio_path = os.path.join(
                self.local_path,
                'generated_data',
                'mixtures_' + subset_name_on_disk + '_' + params_hash,
                'audio'
            )

            mixture_meta_path = os.path.join(
                self.local_path,
                'generated_data',
                'mixtures_' + subset_name_on_disk + '_' + params_hash,
                'meta'
            )

            # Make sure folder exists
            if not os.path.isdir(mixture_path):
                os.makedirs(mixture_path)

            if not os.path.isdir(mixture_audio_path):
                os.makedirs(mixture_audio_path)

            if not os.path.isdir(mixture_meta_path):
                os.makedirs(mixture_meta_path)

            class_progress = tqdm(self.event_labels,
                                  desc="{0: <25s}".format('Class'),
                                  file=sys.stdout,
                                  leave=False,
                                  disable=self.disable_progress_bar,
                                  ascii=self.use_ascii_progress_bar)

            for class_label in class_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{class_label:<30s}]'.format(
                        title='Class ',
                        class_label=class_label)
                    )

                mixture_recipes_filename = os.path.join(
                    mixture_meta_path,
                    'mixture_recipes_' + subset_name_on_disk + '_' + class_label + '.yaml'
                )

                # Generate recipes if not exists
                if not os.path.isfile(mixture_recipes_filename):
                    self._generate_mixture_recipes(
                        params=params,
                        class_label=class_label,
                        subset=subset_name_on_disk,
                        mixture_recipes_filename=mixture_recipes_filename,
                        background_meta=background_meta,
                        event_meta=event_meta[class_label],
                        background_audio_path=background_audio_path,
                        event_audio_path=event_audio_path,
                        r=r
                    )

                mixture_meta = ParameterListFile().load(filename=mixture_recipes_filename)

                # Generate mixture signals
                item_progress = tqdm(mixture_meta,
                                     desc="{0: <25s}".format('Generate mixture'),
                                     file=sys.stdout,
                                     leave=False,
                                     disable=self.disable_progress_bar,
                                     ascii=self.use_ascii_progress_bar)

                for item_id, item in enumerate(item_progress):
                    if self.log_system_progress:
                        self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                            title='Generate mixture ',
                            item_id=item_id,
                            total=len(item_progress),
                            file=item['mixture_audio_filename'])
                        )

                    mixture_file = os.path.join(mixture_audio_path, item['mixture_audio_filename'])

                    if not os.path.isfile(mixture_file):
                        mixture = self._synthesize_mixture(
                            mixture_recipe=item,
                            params=params,
                            background_audio_path=background_audio_path,
                            event_audio_path=event_audio_path
                        )

                        audio_container = AudioFile(
                            data=mixture,
                            fs=params['mixture']['fs']
                        )

                        audio_container.save(
                            filename=mixture_file,
                            bitdepth=params['mixture']['bitdepth']
                        )

                # Generate event lists
                event_list_filename = os.path.join(
                    mixture_meta_path,
                    'event_list_' + subset_name_on_disk + '_' + class_label + '.csv'
                )

                event_list = MetaDataContainer(filename=event_list_filename)
                if not event_list.exists():
                    item_progress = tqdm(mixture_meta,
                                         desc="{0: <25s}".format('Event list'),
                                         file=sys.stdout,
                                         leave=False,
                                         disable=self.disable_progress_bar,
                                         ascii=self.use_ascii_progress_bar)
                    for item_id, item in enumerate(item_progress):
                        if self.log_system_progress:
                            self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                title='Event list ',
                                item_id=item_id,
                                total=len(item_progress),
                                file=item['mixture_audio_filename'])
                            )

                        event_list_item = {
                            'file': os.path.join(
                                'generated_data',
                                'mixtures_' + subset_name_on_disk + '_' + params_hash,
                                'audio',
                                item['mixture_audio_filename']
                            ),
                        }
                        if item['event_present']:
                            event_list_item['event_label'] = item['event_class']
                            event_list_item['event_onset'] = float(item['event_start_in_mixture_seconds'])
                            event_list_item['event_offset'] = float(item['event_start_in_mixture_seconds'] + item['event_length_seconds'])

                        event_list.append(MetaDataItem(event_list_item))
                    event_list.save()

                mixture_parameters = os.path.join(mixture_path, 'parameters.yaml')
                # Save parameters
                if not os.path.isfile(mixture_parameters):
                    ParameterFile(params).save(filename=mixture_parameters)

        if not self.meta_container.exists():
            # Collect meta data
            meta_data = MetaDataContainer()
            for class_label in self.event_labels:

                for subset_label, subset_name_on_disk in iteritems(subset_map):
                    params_hash = self.synth_parameters.get_hash_for_path(subset_label)
                    mixture_meta_path = os.path.join(
                        self.local_path,
                        'generated_data',
                        'mixtures_' + subset_name_on_disk + '_' + params_hash,
                        'meta'
                    )

                    event_list_filename = os.path.join(
                        mixture_meta_path,
                        'event_list_' + subset_name_on_disk + '_' + class_label + '.csv'
                    )

                    meta_data += MetaDataContainer(filename=event_list_filename).load()

            self.meta_container.update(meta_data)
            self.meta_container.save()

        if self.dcase_compatibility:
            # Run dataset patcher if it have not been run before
            if not os.path.exists(self.meta_container.filename+'_old_dontuse'):
                # Create init so we can call functions
                if os.path.exists(os.path.join(self.local_path, 'TUT_Rare_sound_events_mixture_synthesizer', '__init__.py')):
                    open(os.path.join(self.local_path, 'TUT_Rare_sound_events_mixture_synthesizer', '__init__.py'), 'a').close()

                # Patch dataset
                sys.path.append(os.path.join(self.local_path, 'TUT_Rare_sound_events_mixture_synthesizer'))
                from dcase2017_task2_patcher import main as patcher
                patcher(path_to_dataset=self.local_path)

                # Load modified meta information
                self.meta_container.load()

    def _generate_mixture_recipes(self, params, subset, class_label, mixture_recipes_filename, background_meta,
                                  event_meta, background_audio_path, event_audio_path, r):

        def get_event_amplitude_scaling_factor(signal, noise, target_snr_db):
            """Get amplitude scaling factor

            Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
            and rmse is calculated over the whole signal

            Parameters
            ----------
            signal : numpy.ndarray
            noise : numpy.ndarray
            target_snr_db : float

            Returns
            -------
            float > 0.0
            """

            def rmse(y):
                """RMSE"""
                return numpy.sqrt(numpy.mean(numpy.abs(y) ** 2, axis=0, keepdims=False))

            original_sn_rmse_ratio = rmse(signal) / rmse(noise)
            target_sn_rmse_ratio = 10 ** (target_snr_db / float(20))
            signal_scaling_factor = target_sn_rmse_ratio / original_sn_rmse_ratio
            return signal_scaling_factor

        # Internal variables
        fs = float(params.get('mixture').get('fs', 44100))

        current_class_events = []
        # Inject fields to meta data
        for event in event_meta:
            event['classname'] = class_label
            event['audio_filepath'] = os.path.join(class_label, event['audio_filename'])
            event['length_seconds'] = numpy.diff(event['segment'])[0]
            current_class_events.append(event)

        # Randomize order of event and background
        events = r.choice(current_class_events,
                          int(round(params.get('mixtures_per_class') * params.get('event_presence_prob'))))
        bgs = r.choice(background_meta, params.get('mixtures_per_class'))

        # Event presence flags
        event_presence_flags = (numpy.hstack((numpy.ones(len(events)), numpy.zeros(len(bgs) - len(events))))).astype(bool)
        event_presence_flags = r.permutation(event_presence_flags)

        # Event instance IDs, by default event id set to nan: no event. fill it later with actual event ids when needed
        event_instance_ids = numpy.nan * numpy.ones(len(bgs)).astype(int)
        event_instance_ids[event_presence_flags] = numpy.arange(len(events))

        # Randomize event position inside background
        for event in events:
            event['offset_seconds'] = (params.get('mixture').get('length_seconds') - event['length_seconds']) * r.rand()

        # Get offsets for all mixtures, If no event present, use nans
        event_offsets_seconds = numpy.nan * numpy.ones(len(bgs))
        event_offsets_seconds[event_presence_flags] = [event['offset_seconds'] for event in events]

        # Double-check that we didn't shuffle things wrongly: check that the offset never exceeds bg_len-event_len
        checker = []
        for offset, event_instance_id in zip(event_offsets_seconds[event_presence_flags], event_instance_ids[event_presence_flags]):
            checker.append(offset + events[int(event_instance_id)]['length_seconds'])

        assert numpy.max(numpy.array(checker)) < params.get('mixture').get('length_seconds')

        # Target EBRs
        target_ebrs = -numpy.inf * numpy.ones(len(bgs))
        target_ebrs[event_presence_flags] = r.choice(params.get('ebr_list'), size=numpy.sum(event_presence_flags))

        # For recipes, we got to provide amplitude scaling factors instead of SNRs: the latter are more ambiguous
        # so, go through files, measure levels, calculate scaling factors
        mixture_recipes = ParameterListFile()
        for mixture_id, (bg, event_presence_flag, event_start_in_mixture_seconds, ebr, event_instance_id) in tqdm(
                enumerate(zip(bgs, event_presence_flags, event_offsets_seconds, target_ebrs, event_instance_ids)),
                desc="{0: <25s}".format('Generate recipe'),
                file=sys.stdout,
                leave=False,
                total=len(bgs),
                disable=self.disable_progress_bar,
                ascii=self.use_ascii_progress_bar):

            # Read the bgs and events, measure their energies, find amplitude scaling factors
            mixture_recipe = {
                'bg_path': bg['filepath'],
                'bg_classname': bg['classname'],
                'event_present': bool(event_presence_flag),
                'ebr': float(ebr)
            }

            if event_presence_flag:
                # We have an event assigned

                assert not numpy.isnan(event_instance_id)

                # Load background and event audio in
                bg_audio, fs_bg = AudioFile(fs=params.get('mixture').get('fs')).load(
                    filename=os.path.join(background_audio_path, bg['filepath'])
                )

                event_audio, fs_event = AudioFile(fs=params.get('mixture').get('fs')).load(
                    filename=os.path.join(event_audio_path, events[int(event_instance_id)]['audio_filepath'])
                )

                assert fs_bg == fs_event, 'Fs mismatch! Expected resampling taken place already'

                # Segment onset and offset in samples
                segment_start_samples = int(events[int(event_instance_id)]['segment'][0] * fs)
                segment_end_samples = int(events[int(event_instance_id)]['segment'][1] * fs)

                # Cut event audio
                event_audio = event_audio[segment_start_samples:segment_end_samples]

                # Let's calculate the levels of bgs also at the location of the event only
                eventful_part_of_bg = bg_audio[int(event_start_in_mixture_seconds * fs):int(event_start_in_mixture_seconds * fs + len(event_audio))]

                if eventful_part_of_bg.shape[0] == 0:
                    message = '{name}: Background segment having an event has zero length.'.format(
                        name=self.__class__.__name__
                    )
                    self.logger.exception(message)
                    raise ValueError(message)

                scaling_factor = get_event_amplitude_scaling_factor(event_audio, eventful_part_of_bg, target_snr_db=ebr)

                # Store information
                mixture_recipe['event_path'] = events[int(event_instance_id)]['audio_filepath']
                mixture_recipe['event_class'] = events[int(event_instance_id)]['classname']
                mixture_recipe['event_start_in_mixture_seconds'] = float(event_start_in_mixture_seconds)
                mixture_recipe['event_length_seconds'] = float(events[int(event_instance_id)]['length_seconds'])
                mixture_recipe['scaling_factor'] = float(scaling_factor)
                mixture_recipe['segment_start_seconds'] = events[int(event_instance_id)]['segment'][0]
                mixture_recipe['segment_end_seconds'] = events[int(event_instance_id)]['segment'][1]

            # Generate mixture filename
            mixing_param_hash = hashlib.md5(yaml.dump(mixture_recipe).encode('utf-8')).hexdigest()
            mixture_recipe['mixture_audio_filename'] = 'mixture' + '_' + subset + '_' + class_label + '_' + '%03d' % mixture_id + '_' + mixing_param_hash + '.' + self.default_audio_extension

            # Generate mixture annotation
            if event_presence_flag:
                mixture_recipe['annotation_string'] = \
                    mixture_recipe['mixture_audio_filename'] + '\t' + \
                    "{0:.14f}".format(mixture_recipe['event_start_in_mixture_seconds']) + '\t' + \
                    "{0:.14f}".format(mixture_recipe['event_start_in_mixture_seconds'] + mixture_recipe['event_length_seconds']) + '\t' + \
                    mixture_recipe['event_class']

            else:
                mixture_recipe['annotation_string'] = mixture_recipe['mixture_audio_filename'] + '\t' + 'None' + '\t0\t30'

            # Store mixture recipe
            mixture_recipes.append(mixture_recipe)

        # Save mixture recipe
        mixture_recipes.save(filename=mixture_recipes_filename)

    def _synthesize_mixture(self, mixture_recipe, params, background_audio_path, event_audio_path):
        background_audiofile = os.path.join(background_audio_path, mixture_recipe['bg_path'])

        # Load background audio
        bg_audio_data, fs_bg = AudioFile().load(filename=background_audiofile,
                                                fs=params['mixture']['fs'],
                                                mono=True)

        if mixture_recipe['event_present']:
            event_audiofile = os.path.join(event_audio_path, mixture_recipe['event_path'])

            # Load event audio
            event_audio_data, fs_event = AudioFile().load(filename=event_audiofile,
                                                          fs=params['mixture']['fs'],
                                                          mono=True)

            if fs_bg != fs_event:
                message = '{name}: Sampling frequency mismatch. Material should be resampled.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise ValueError(message)

            # Slice event audio
            segment_start_samples = int(mixture_recipe['segment_start_seconds'] * params['mixture']['fs'])
            segment_end_samples = int(mixture_recipe['segment_end_seconds'] * params['mixture']['fs'])
            event_audio_data = event_audio_data[segment_start_samples:segment_end_samples]

            event_start_in_mixture_samples = int(mixture_recipe['event_start_in_mixture_seconds'] * params['mixture']['fs'])
            scaling_factor = mixture_recipe['scaling_factor']

            # Mix event into background audio
            mixture = self._mix(bg_audio_data=bg_audio_data,
                                event_audio_data=event_audio_data,
                                event_start_in_mixture_samples=event_start_in_mixture_samples,
                                scaling_factor=scaling_factor,
                                magic_anticlipping_factor=params['mixture']['anticlipping_factor'])
        else:
            mixture = params['mixture']['anticlipping_factor'] * bg_audio_data

        return mixture

    def _mix(self, bg_audio_data, event_audio_data, event_start_in_mixture_samples, scaling_factor, magic_anticlipping_factor):
        """Mix numpy arrays of background and event audio (mono, non-matching lengths supported, sampling frequency
        better be the same, no operation in terms of seconds is performed though)

        Parameters
        ----------
        bg_audio_data : numpy.array
        event_audio_data : numpy.array
        event_start_in_mixture_samples : float
        scaling_factor : float
        magic_anticlipping_factor : float

        Returns
        -------
        numpy.array

        """

        # Store current event audio max value
        event_audio_original_max = numpy.max(numpy.abs(event_audio_data))

        # Adjust SNRs
        event_audio_data *= scaling_factor

        # Check that the offset is not too long
        longest_possible_offset = len(bg_audio_data) - len(event_audio_data)
        if event_start_in_mixture_samples > longest_possible_offset:
            message = '{name}: Wrongly generated event offset: event tries to go outside the boundaries of the bg.'.format(name=self.__class__.__name__)
            self.logger.exception(message)
            raise AssertionError(message)

        # Measure how much to pad from the right
        tail_length = len(bg_audio_data) - len(event_audio_data) - event_start_in_mixture_samples

        # Pad zeros at the beginning of event signal
        padded_event = numpy.pad(event_audio_data,
                                 pad_width=((event_start_in_mixture_samples, tail_length)),
                                 mode='constant',
                                 constant_values=0)

        if not len(padded_event) == len(bg_audio_data):
            message = '{name}: Mixing yielded a signal of different length than bg! Should not happen.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise AssertionError(message)

        mixture = magic_anticlipping_factor * (padded_event + bg_audio_data)

        # Also nice to make sure that we did not introduce clipping
        if numpy.max(numpy.abs(mixture)) >= 1:
            normalisation_factor = 1 / float(numpy.max(numpy.abs(mixture)))
            print('Attention! Had to normalise the mixture by [{factor}]'.format(factor=normalisation_factor))
            print('I.e. bg max: {bg_max:2.4f}, event max: {event_max:2.4f}, sum max: {sum_max:2.4f}'.format(
                bg_max=numpy.max(numpy.abs(bg_audio_data)),
                event_max=numpy.max(numpy.abs(padded_event)),
                sum_max=numpy.max(numpy.abs(mixture)))
            )
            print('The scaling factor for the event was [{factor}]'.format(factor=scaling_factor))
            print('The event before scaling was max [{max}]'.format(max=event_audio_original_max))
            mixture /= numpy.max(numpy.abs(mixture))

        return mixture


class TUTRareSoundEvents_2017_EvaluationSet(SyntheticSoundEventDataset):
    """TUT Acoustic scenes 2017 evaluation dataset

    This dataset is used in DCASE2017 - Task 2, Rare sound event detection

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-rare-sound-events-2017-evaluation')
        kwargs['filelisthash_exclude_dirs'] = kwargs.get('filelisthash_exclude_dirs', ['generated_data'])

        # Initialize baseclass
        super(TUTRareSoundEvents_2017_EvaluationSet, self).__init__(*args, **kwargs)

        self.reference_data_present = False

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Aleksandr Diment, Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Rare Sound Events 2017, evaluation dataset',
            'url': None,
            'audio_source': 'Synthetic',
            'audio_type': 'Natural',
            'recording_device_model': 'Unknown',
            'microphone_model': 'Unknown',
        }

        self.crossvalidation_folds = 1

        source_url = 'https://zenodo.org/record/1160455/files/'

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-rare-sound-events-2017-evaluation.mixture_data.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-evaluation.mixture_data.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-rare-sound-events-2017-evaluation.mixture_data.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-evaluation.mixture_data.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-rare-sound-events-2017-evaluation.mixture_data.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-evaluation.mixture_data.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-rare-sound-events-2017-evaluation.mixture_data.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-evaluation.mixture_data.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-rare-sound-events-2017-evaluation.mixture_data.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-rare-sound-events-2017-evaluation.mixture_data.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'data', 'mixture_data', 'evaltest', 'bbb81504db15a03680a0044474633b67', 'audio'),
            },
        ]

    @property
    def event_labels(self, scene_label=None):
        """List of unique event labels in the meta data.

        Parameters
        ----------

        Returns
        -------
        labels : list
            List of event labels in alphabetical order.

        """

        labels = ['babycry', 'glassbreak', 'gunshot']
        labels.sort()
        return labels

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists() and self.reference_data_present:
            message = '{name}: Meta file not found [{filename}]'.format(
                name=self.__class__.__name__,
                filename=self.meta_container.filename
            )

            self.logger.exception(message)
            raise IOError(message)

    def train(self, fold=0, scene_label=None, event_label=None):
        return []

    def test(self, fold=0, scene_label=None, event_label=None):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        event_label : str
            Event label
            Default value "None"

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for event_label_ in self.event_labels:
                if event_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][event_label_] = MetaDataContainer()

                if fold == 0:
                    for filename in self.audio_files:
                        raw_path, raw_filename = os.path.split(filename)
                        relative_path = self.absolute_to_relative(raw_path)
                        base_filename, file_extension = os.path.splitext(raw_filename)
                        if event_label_ in base_filename:
                            self.crossvalidation_data_test[fold][event_label_].append(MetaDataItem({'file': os.path.join(relative_path, raw_filename)}))

        if event_label:
            return self.crossvalidation_data_test[fold][event_label]
        else:
            data = MetaDataContainer()
            for event_label_ in self.event_labels:
                data += self.crossvalidation_data_test[fold][event_label_]

            return data


class TUTSoundEvents_2017_DevelopmentSet(SoundEventDataset):
    """TUT Sound events 2017 development dataset

    This dataset is used in DCASE2017 - Task 3, Sound event detection in real life audio

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-sound-events-2017-development')
        super(TUTSoundEvents_2017_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Sound Events 2016, development dataset',
            'url': 'https://zenodo.org/record/45759',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 4

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'street'),
            },
            {
                'remote_package': 'https://zenodo.org/record/814831/files/TUT-sound-events-2017-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/814831/files/TUT-sound-events-2017-development.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-development.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/814831/files/TUT-sound-events-2017-development.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-development.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/814831/files/TUT-sound-events-2017-development.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-development.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            meta_data = MetaDataContainer()
            for filename in self.audio_files:
                raw_path, raw_filename = os.path.split(filename)
                relative_path = self.absolute_to_relative(raw_path)
                scene_label = relative_path.replace('audio', '')[1:]
                base_filename, file_extension = os.path.splitext(raw_filename)
                annotation_filename = os.path.join(
                    self.local_path,
                    relative_path.replace('audio', 'meta'),
                    base_filename + '.ann'
                )

                data = MetaDataContainer(filename=annotation_filename).load()
                for item in data:
                    item['file'] = os.path.join(relative_path, raw_filename)
                    item['scene_label'] = scene_label
                    item['identifier'] = os.path.splitext(raw_filename)[0]
                    item['source_label'] = 'mixture'

                meta_data += data
            self.meta_container.update(meta_data)
            self.meta_container.save()
        else:
            self.meta_container.load()

    def train(self, fold=0, scene_label=None):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            Default value "None"

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_train[fold]:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer()

                if fold > 0:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='train',
                            fold=fold, scene_label=scene_label_)).load()

                else:
                    self.crossvalidation_data_train[0][scene_label_] = self.meta_container.filter(
                        scene_label=scene_label_
                    )

                for item in self.crossvalidation_data_train[fold][scene_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])
                    raw_path, raw_filename = os.path.split(item['file'])
                    item['identifier'] = os.path.splitext(raw_filename)[0]
                    item['source_label'] = 'mixture'

        if scene_label:
            return self.crossvalidation_data_train[fold][scene_label]
        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_train[fold][scene_label_]

            return data


class TUTSoundEvents_2017_EvaluationSet(SoundEventDataset):
    """TUT Sound events 2017 evaluation dataset

    This dataset is used in DCASE2017 - Task 3, Sound event detection in real life audio

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-sound-events-2017-evaluation')
        super(TUTSoundEvents_2017_EvaluationSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Sound Events 2016, development dataset',
            'url': 'https://zenodo.org/record/45759',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 1
        source_url = 'https://zenodo.org/record/1040179/files/'
        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'street'),
            },
            {
                'remote_package': source_url + 'TUT-sound-events-2017-evaluation.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-evaluation.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-sound-events-2017-evaluation.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-evaluation.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': source_url + 'TUT-sound-events-2017-evaluation.audio.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2017-evaluation.audio.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
        ]

    @property
    def scene_labels(self):
        labels = ['street']
        labels.sort()
        return labels

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if self.reference_data_present:
            if not self.meta_container.exists():
                meta_data = MetaDataContainer()
                for filename in self.audio_files:
                    raw_path, raw_filename = os.path.split(filename)
                    relative_path = self.absolute_to_relative(raw_path)
                    scene_label = relative_path.replace('audio', '')[1:]
                    base_filename, file_extension = os.path.splitext(raw_filename)
                    annotation_filename = os.path.join(self.local_path, relative_path.replace('audio', 'meta'),
                                                       base_filename + '.ann')
                    data = MetaDataContainer(filename=annotation_filename).load()
                    for item in data:
                        item['file'] = os.path.join(relative_path, raw_filename)
                        item['scene_label'] = scene_label
                        item['identifier'] = os.path.splitext(raw_filename)[0]
                        item['source_label'] = 'mixture'

                    meta_data += data
                meta_data.save(filename=self.meta_container.filename)
                self.meta_container.load()

            else:
                self.meta_container.load()

    def train(self, fold=0, scene_label=None):
        return []

    def test(self, fold=0, scene_label=None):
        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer()

                if fold > 0:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='test', fold=fold, scene_label=scene_label_)
                    ).load()
                else:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='test', fold=fold, scene_label=scene_label_)
                    ).load()

        if scene_label:
            return self.crossvalidation_data_test[fold][scene_label]
        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_test[fold][scene_label_]

            return data


class DCASE2017_Task4tagging_DevelopmentSet(SoundEventDataset):
    """DCASE 2017 Large-scale weakly supervised sound event detection for smart cars

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'DCASE2017-task4-development')
        super(DCASE2017_Task4tagging_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'audio tagging'
        self.dataset_meta = {
            'authors': 'Benjamin Elizalde, Rohan Badlani, Ankit Shah, Emmanuel Vincent, Bhiksha Raj',
            'name_remote': 'Task 4 Large-scale weakly supervised sound event detection for smart cars',
            'url': 'https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': None,
            'microphone_model': None,
        }

        self.crossvalidation_folds = 1
        self.default_audio_extension = 'flac'

        github_url = 'https://raw.githubusercontent.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/master/'

        self.package_list = [
            {
                'remote_package': github_url + 'training_set.csv',
                'local_package': os.path.join(self.local_path, 'training_set.csv'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'testing_set.csv',
                'local_package': os.path.join(self.local_path, 'testing_set.csv'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'groundtruth_weak_label_training_set.csv',
                'local_package': os.path.join(self.local_path, 'groundtruth_weak_label_training_set.csv'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'groundtruth_weak_label_testing_set.csv',
                'local_package': os.path.join(self.local_path, 'groundtruth_weak_label_testing_set.csv'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'APACHE_LICENSE.txt',
                'local_package': os.path.join(self.local_path, 'APACHE_LICENSE.txt'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'README.txt',
                'local_package': os.path.join(self.local_path, 'README.txt'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'sound_event_list_17_classes.txt',
                'local_package': os.path.join(self.local_path, 'sound_event_list_17_classes.txt'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': github_url + 'groundtruth_strong_label_testing_set.csv',
                'local_package': os.path.join(self.local_path, 'groundtruth_strong_label_testing_set.csv'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    @property
    def scene_labels(self):
        labels = ['youtube']
        labels.sort()
        return labels

    def _after_extract(self, to_return=None):
        import csv
        from httplib import BadStatusLine
        from dcase_framework.files import AudioFile

        def progress_hook(t):
            """
            Wraps tqdm instance. Don't forget to close() or __exit__()
            the tqdm instance once you're done with it (easiest using `with` syntax).
            """

            def inner(total, recvd, ratio, rate, eta):
                t.total = int(total / 1024.0)
                t.update(int(recvd / 1024.0))

            return inner

        # Collect file ids
        files = []
        with open(os.path.join(self.local_path, 'testing_set.csv'), 'rb') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                files.append({
                    'query_id': row[0],
                    'segment_start': row[1],
                    'segment_end': row[2]}
                )

        with open(os.path.join(self.local_path, 'training_set.csv'), 'rb') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                files.append({
                    'query_id': row[0],
                    'segment_start': row[1],
                    'segment_end': row[2]}
                )

        # Make sure audio directory exists
        if not os.path.isdir(os.path.join(self.local_path, 'audio')):
            os.makedirs(os.path.join(self.local_path, 'audio'))

        file_progress = tqdm(files,
                             desc="{0: <25s}".format('Files'),
                             file=sys.stdout,
                             leave=False,
                             disable=self.disable_progress_bar,
                             ascii=self.use_ascii_progress_bar)

        non_existing_videos = []

        # Check that audio files exists
        for file_data in file_progress:
            audio_filename = os.path.join(self.local_path,
                                          'audio',
                                          'Y{query_id}_{segment_start}_{segment_end}.{extension}'.format(
                                              query_id=file_data['query_id'],
                                              segment_start=file_data['segment_start'],
                                              segment_end=file_data['segment_end'],
                                              extension=self.default_audio_extension
                                          )
                                          )

            # Download segment if it does not exists
            if not os.path.isfile(audio_filename):
                import pafy

                #
                try:
                    # Access youtube video and get best quality audio stream
                    youtube_audio = pafy.new(
                        url='https://www.youtube.com/watch?v={query_id}'.format(query_id=file_data['query_id']),
                        basic=False,
                        gdata=False,
                        size=False
                    ).getbestaudio()

                    # Get temp file
                    tmp_file = os.path.join(self.local_path, 'tmp_file.{extension}'.format(
                        extension=youtube_audio.extension)
                                            )

                    # Create download progress bar
                    download_progress_bar = tqdm(
                        desc="{0: <25s}".format('Download youtube item '),
                        file=sys.stdout,
                        unit='B',
                        unit_scale=True,
                        leave=False,
                        disable=self.disable_progress_bar,
                        ascii=self.use_ascii_progress_bar
                    )

                    # Download audio
                    youtube_audio.download(
                        filepath=tmp_file,
                        quiet=True,
                        callback=progress_hook(download_progress_bar)
                    )

                    # Close progress bar
                    download_progress_bar.close()

                    # Create audio processing progress bar
                    audio_processing_progress_bar = tqdm(
                        desc="{0: <25s}".format('Processing '),
                        initial=0,
                        total=4,
                        file=sys.stdout,
                        leave=False,
                        disable=self.disable_progress_bar,
                        ascii=self.use_ascii_progress_bar
                    )

                    # Load audio
                    audio_file = AudioFile()
                    audio_file.load(
                        filename=tmp_file,
                        mono=True,
                        fs=44100,
                        res_type='kaiser_best',
                        start=float(file_data['segment_start']),
                        stop=float(file_data['segment_end'])
                    )
                    audio_processing_progress_bar.update(1)

                    # Save the segment
                    audio_file.save(
                        filename=audio_filename,
                        bitdepth=16
                    )
                    audio_processing_progress_bar.update(3)

                    # Remove temporal file
                    os.remove(tmp_file)
                    audio_processing_progress_bar.close()

                except (IOError, BadStatusLine) as e:
                    # Store files with errors
                    file_data['error'] = str(e.message)
                    non_existing_videos.append(file_data)

                except (KeyboardInterrupt, SystemExit):
                    # Remove temporal file and current audio file.
                    os.remove(tmp_file)
                    os.remove(audio_filename)
                    raise

        log_filename = os.path.join(self.local_path, 'item_access_error.log')
        with open(log_filename, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for item in non_existing_videos:
                csv_writer.writerow(
                    (item['query_id'], item['error'].replace('\n', ' '))
                )

        # Make sure evaluation_setup directory exists
        if not os.path.isdir(os.path.join(self.local_path, self.evaluation_setup_folder)):
            os.makedirs(os.path.join(self.local_path, self.evaluation_setup_folder))

        # Check that evaluation setup exists
        evaluation_setup_exists = True
        train_filename = self._get_evaluation_setup_filename(
            setup_part='train',
            fold=1,
            scene_label='youtube',
            file_extension='txt'
        )

        test_filename = self._get_evaluation_setup_filename(
            setup_part='test',
            fold=1,
            scene_label='youtube',
            file_extension='txt'
        )

        evaluate_filename = self._get_evaluation_setup_filename(
            setup_part='evaluate',
            fold=1,
            scene_label='youtube',
            file_extension='txt'
        )

        if not os.path.isfile(train_filename) or not os.path.isfile(test_filename) or not os.path.isfile(
                evaluate_filename):
            evaluation_setup_exists = False

        # Evaluation setup was not found generate
        if not evaluation_setup_exists:
            fold = 1

            train_meta = MetaDataContainer()
            for item in MetaDataContainer().load(
                    os.path.join(self.local_path, 'groundtruth_weak_label_training_set.csv')):
                if not item['file'].endswith('flac'):
                    item['file'] = os.path.join('audio', 'Y' + os.path.splitext(item['file'])[
                        0] + '.' + self.default_audio_extension)
                # Set scene label
                item['scene_label'] = 'youtube'

                # Translate event onset and offset, weak labels
                item['event_offset'] -= item['event_onset']
                item['event_onset'] -= item['event_onset']

                # Only collect items which exists
                if os.path.isfile(os.path.join(self.local_path, item['file'])):
                    train_meta.append(item)

            train_meta.save(filename=self._get_evaluation_setup_filename(
                setup_part='train',
                fold=fold,
                scene_label='youtube',
                file_extension='txt')
            )

            evaluate_meta = MetaDataContainer()
            for item in MetaDataContainer().load(
                    os.path.join(self.local_path, 'groundtruth_strong_label_testing_set.csv')):
                if not item['file'].endswith('flac'):
                    item['file'] = os.path.join('audio', 'Y' + os.path.splitext(item['file'])[
                        0] + '.' + self.default_audio_extension)
                # Set scene label
                item['scene_label'] = 'youtube'

                # Only collect items which exists
                if os.path.isfile(os.path.join(self.local_path, item['file'])):
                    evaluate_meta.append(item)

            evaluate_meta.save(filename=self._get_evaluation_setup_filename(
                setup_part='evaluate',
                fold=fold,
                scene_label='youtube',
                file_extension='txt')
            )

            test_meta = MetaDataContainer()
            for item in evaluate_meta:
                test_meta.append(MetaDataItem({'file': item['file']}))

                test_meta.save(filename=self._get_evaluation_setup_filename(
                    setup_part='test',
                    fold=fold,
                    scene_label='youtube',
                    file_extension='txt')
                )

        if not self.meta_container.exists():
            fold = 1
            meta_data = MetaDataContainer()
            meta_data += MetaDataContainer().load(self._get_evaluation_setup_filename(
                setup_part='train',
                fold=fold,
                scene_label='youtube',
                file_extension='txt')
            )

            meta_data += MetaDataContainer().load(self._get_evaluation_setup_filename(
                setup_part='evaluate',
                fold=fold,
                scene_label='youtube',
                file_extension='txt')
            )

            self.meta_container.update(meta_data)
            self.meta_container.save()
        else:
            self.meta_container.load()


class DCASE2017_Task4tagging_EvaluationSet(DCASE2017_Task4tagging_DevelopmentSet):
    """DCASE 2017 Large-scale weakly supervised sound event detection for smart cars

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'DCASE2017-task4-evaluation')
        super(DCASE2017_Task4tagging_DevelopmentSet, self).__init__(*args, **kwargs)

        self.reference_data_present = False

        self.dataset_group = 'audio tagging'
        self.dataset_meta = {
            'authors': 'Benjamin Elizalde, Rohan Badlani, Ankit Shah, Emmanuel Vincent, Bhiksha Raj',
            'name_remote': 'Task 4 Large-scale weakly supervised sound event detection for smart cars',
            'url': 'https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': None,
            'microphone_model': None,
        }

        self.crossvalidation_folds = 1
        self.default_audio_extension = 'wav'

        self.package_list = [
            {
                'remote_package': 'https://dl.dropboxusercontent.com/s/bbgqfd47cudwe9y/DCASE_2017_evaluation_set_audio_files.zip',
                'local_package': os.path.join(self.local_path, 'DCASE_2017_evaluation_set_audio_files.zip'),
                'local_audio_path': os.path.join(self.local_path),
                'zip_password': 'DCASE_2017_evaluation_set',
            }
        ]

    def _after_extract(self, to_return=None):
        if not os.path.exists(os.path.join(self.local_path, 'audio')):
            os.makedirs(os.path.join(self.local_path, 'audio'))

        # Get audio files at the root directory
        audio_files = []
        l = os.listdir(self.local_path)
        for f in l:
            file_name, file_extension = os.path.splitext(f)
            if file_extension[1:] in self.audio_extensions:
                if os.path.abspath(os.path.join(self.local_path, f)) not in audio_files:
                    audio_files.append(os.path.abspath(os.path.join(self.local_path, f)))

        reference_data_file = os.path.join(self.local_path, 'groundtruth_strong_label_evaluation_set.csv')
        if not self.meta_container.exists() and os.path.exists(reference_data_file):
            # Reference data is prensent and but meta data is empty
            meta_data = MetaDataContainer()
            ref_data = MetaDataContainer().load(filename=reference_data_file)
            for item in ref_data:
                # Modify audio file path
                item['file'] = os.path.join('Y' + os.path.splitext(item['file'])[0] + '.' + self.default_audio_extension)

                # Set scene label
                item['scene_label'] = 'youtube'

                # Only collect items which exists
                if os.path.isfile(os.path.join(self.local_path, item['file'])):
                    meta_data.append(item)
            # Save meta data
            meta_data.save(filename=self.meta_container.filename)

        # Load meta data if present
        if self.meta_container.exists():
            self.meta_container.load()

    def train(self, fold=0, scene_label=None):
        return []

    def test(self, fold=0, scene_label=None):
        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][scene_label_] = MetaDataContainer()

                if fold == 0:
                    for filename in self.audio_files:
                        raw_path, raw_filename = os.path.split(filename)
                        relative_path = self.absolute_to_relative(raw_path)
                        self.crossvalidation_data_test[fold][scene_label_].append(MetaDataItem({'file': os.path.join(relative_path, raw_filename)}))

        if scene_label:
            return self.crossvalidation_data_test[fold][scene_label]
        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_test[fold][scene_label_]

            return data

# =====================================================
# DCASE 2016
# =====================================================
class TUTAcousticScenes_2016_DevelopmentSet(AcousticSceneDataset):
    """TUT Acoustic scenes 2016 development dataset

    This dataset is used in DCASE2016 - Task 1, Acoustic scene classification

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-acoustic-scenes-2016-development')
        super(TUTAcousticScenes_2016_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'acoustic scene'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Acoustic Scenes 2016, development dataset',
            'url': 'https://zenodo.org/record/45739',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 4

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.error.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.error.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.6.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.6.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.7.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.7.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.8.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.8.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """
        if not self.meta_container.exists():
            meta_data = {}
            for fold in range(1, self.crossvalidation_folds):
                # Read train files in
                fold_data = MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt')).load()
                fold_data += MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_evaluate.txt')).load()
                for item in fold_data:
                    if item['file'] not in meta_data:
                        raw_path, raw_filename = os.path.split(item['file'])
                        relative_path = self.absolute_to_relative(raw_path)
                        location_id = raw_filename.split('_')[0]
                        item['file'] = os.path.join(relative_path, raw_filename)
                        item['identifier'] = location_id
                        meta_data[item['file']] = item

            self.meta_container.update(meta_data.values())
            self.meta_container.save()

    def train(self, fold=0):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = []
            if fold > 0:
                self.crossvalidation_data_train[fold] = MetaDataContainer(
                    filename=os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt')).load()
                for item in self.crossvalidation_data_train[fold]:
                    item['file'] = self.relative_to_absolute_path(item['file'])
                    raw_path, raw_filename = os.path.split(item['file'])
                    location_id = raw_filename.split('_')[0]
                    item['identifier'] = location_id
            else:
                self.crossvalidation_data_train[0] = self.meta_container

        return self.crossvalidation_data_train[fold]


class TUTAcousticScenes_2016_EvaluationSet(AcousticSceneDataset):
    """TUT Acoustic scenes 2016 evaluation dataset

    This dataset is used in DCASE2016 - Task 1, Acoustic scene classification

    """
    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-acoustic-scenes-2016-evaluation')
        super(TUTAcousticScenes_2016_EvaluationSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'acoustic scene'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Acoustic Scenes 2016, evaluation dataset',
            'url': 'https://zenodo.org/record/165995',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 1

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        eval_file = MetaDataContainer(filename=os.path.join(self.evaluation_setup_path, 'evaluate.txt'))
        if not self.meta_container.exists() and eval_file.exists():
            eval_data = eval_file.load()
            meta_data = {}
            for item in eval_data:
                if item['file'] not in meta_data:
                    raw_path, raw_filename = os.path.split(item['file'])
                    relative_path = self.absolute_to_relative(raw_path)
                    item['file'] = os.path.join(relative_path, raw_filename)
                    meta_data[item['file']] = item

            self.meta_container.update(meta_data.values())
            self.meta_container.save()

    def train(self, fold=0):
        return []

    def test(self, fold=0):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.crossvalidation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
            else:
                data = []
                files = []
                for item in self.audio_files:
                    if self.relative_to_absolute_path(item) not in files:
                        data.append({'file': self.relative_to_absolute_path(item)})
                        files.append(self.relative_to_absolute_path(item))

                self.crossvalidation_data_test[fold] = data

        return self.crossvalidation_data_test[fold]


class TUTSoundEvents_2016_DevelopmentSet(SoundEventDataset):
    """TUT Sound events 2016 development dataset

    This dataset is used in DCASE2016 - Task 3, Sound event detection in real life audio

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-sound-events-2016-development')
        super(TUTSoundEvents_2016_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Sound Events 2016, development dataset',
            'url': 'https://zenodo.org/record/45759',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 4

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'residential_area'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'home'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-development.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.audio.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-development.audio.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            meta_data = MetaDataContainer()
            for filename in self.audio_files:
                raw_path, raw_filename = os.path.split(filename)
                relative_path = self.absolute_to_relative(raw_path)
                scene_label = relative_path.replace('audio', '')[1:]
                base_filename, file_extension = os.path.splitext(raw_filename)
                annotation_filename = os.path.join(
                    self.local_path,
                    relative_path.replace('audio', 'meta'),
                    base_filename + '.ann'
                )

                data = MetaDataContainer(filename=annotation_filename).load()
                for item in data:
                    item['file'] = os.path.join(relative_path, raw_filename)
                    item['scene_label'] = scene_label
                    item['identifier'] = os.path.splitext(raw_filename)[0]
                    item['source_label'] = 'mixture'

                meta_data += data
            meta_data.save(filename=self.meta_container.filename)

    def train(self, fold=0, scene_label=None):
        """List of training items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)
        scene_label : str
            Scene label
            Default value "None"

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to training set for given fold.

        """

        if fold not in self.crossvalidation_data_train:
            self.crossvalidation_data_train[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_train[fold]:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer()

                if fold > 0:
                    self.crossvalidation_data_train[fold][scene_label_] = MetaDataContainer(
                        filename=self._get_evaluation_setup_filename(
                            setup_part='train', fold=fold, scene_label=scene_label_)).load()
                else:
                    self.crossvalidation_data_train[0][scene_label_] = self.meta_container.filter(
                        scene_label=scene_label_
                    )

                for item in self.crossvalidation_data_train[fold][scene_label_]:
                    item['file'] = self.relative_to_absolute_path(item['file'])
                    raw_path, raw_filename = os.path.split(item['file'])
                    item['identifier'] = os.path.splitext(raw_filename)[0]
                    item['source_label'] = 'mixture'

        if scene_label:
            return self.crossvalidation_data_train[fold][scene_label]
        else:
            data = MetaDataContainer()
            for scene_label_ in self.scene_labels:
                data += self.crossvalidation_data_train[fold][scene_label_]

            return data


class TUTSoundEvents_2016_EvaluationSet(SoundEventDataset):
    """TUT Sound events 2016 evaluation dataset

    This dataset is used in DCASE2016 - Task 3, Sound event detection in real life audio

    """
    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-sound-events-2016-evaluation')
        super(TUTSoundEvents_2016_EvaluationSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen',
            'name_remote': 'TUT Sound Events 2016, evaluation dataset',
            'url': 'http://www.cs.tut.fi/sgn/arg/dcase2016/download/',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Roland Edirol R-09',
            'microphone_model': 'Soundman OKM II Klassik/studio A3 electret microphone',
        }

        self.crossvalidation_folds = 1

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'home'),
            },
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio', 'residential_area'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/dcase2016/evaluation_data/TUT-sound-events-2016-evaluation.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-evaluation.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/dcase2016/evaluation_data/TUT-sound-events-2016-evaluation.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-evaluation.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/dcase2016/evaluation_data/TUT-sound-events-2016-evaluation.audio.zip',
                'local_package': os.path.join(self.local_path, 'TUT-sound-events-2016-evaluation.audio.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },

        ]

    @property
    def scene_labels(self):
        labels = ['home', 'residential_area']
        labels.sort()
        return labels

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists() and os.path.isdir(os.path.join(self.local_path, 'meta')):
            meta_file_handle = open(self.meta_container.filename, 'wt')
            try:
                writer = csv.writer(meta_file_handle, delimiter='\t')
                for filename in self.audio_files:
                    raw_path, raw_filename = os.path.split(filename)
                    relative_path = self.absolute_to_relative(raw_path)
                    scene_label = relative_path.replace('audio', '')[1:]
                    base_filename, file_extension = os.path.splitext(raw_filename)

                    annotation_filename = os.path.join(
                        self.local_path,
                        relative_path.replace('audio', 'meta'),
                        base_filename + '.ann'
                    )

                    if os.path.isfile(annotation_filename):
                        annotation_file_handle = open(annotation_filename, 'rt')
                        try:
                            annotation_file_reader = csv.reader(annotation_file_handle, delimiter='\t')
                            for annotation_file_row in annotation_file_reader:
                                writer.writerow((os.path.join(relative_path, raw_filename),
                                                 scene_label,
                                                 float(annotation_file_row[0].replace(',', '.')),
                                                 float(annotation_file_row[1].replace(',', '.')),
                                                 annotation_file_row[2], 'm'))
                        finally:
                            annotation_file_handle.close()
            finally:
                meta_file_handle.close()

    def train(self, fold=0, scene_label=None):
        return []

    def test(self, fold=0, scene_label=None):
        if fold not in self.crossvalidation_data_test:
            self.crossvalidation_data_test[fold] = {}
            for scene_label_ in self.scene_labels:
                if scene_label_ not in self.crossvalidation_data_test[fold]:
                    self.crossvalidation_data_test[fold][scene_label_] = []

                if fold > 0:
                    with open(
                            os.path.join(self.evaluation_setup_path, scene_label_ + '_fold' + str(fold) + '_test.txt'),
                            'rt') as f:

                        for row in csv.reader(f, delimiter='\t'):
                            self.crossvalidation_data_test[fold][scene_label_].append(
                                {'file': self.relative_to_absolute_path(row[0])}
                            )
                else:
                    with open(os.path.join(self.evaluation_setup_path, scene_label_ + '_test.txt'), 'rt') as f:
                        for row in csv.reader(f, delimiter='\t'):
                            self.crossvalidation_data_test[fold][scene_label_].append(
                                {'file': self.relative_to_absolute_path(row[0])}
                            )

        if scene_label:
            return self.crossvalidation_data_test[fold][scene_label]
        else:
            data = []
            for scene_label_ in self.scene_labels:
                for item in self.crossvalidation_data_test[fold][scene_label_]:
                    data.append(item)
            return data


class CHiMEHome_DomesticAudioTag_DevelopmentSet(AudioTaggingDataset):
    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'CHiMeHome-audiotag-development')
        super(CHiMEHome_DomesticAudioTag_DevelopmentSet, self).__init__(*args, **kwargs)

        self.dataset_group = 'audio tagging'
        self.dataset_meta = {
            'authors': 'Peter Foster, Siddharth Sigtia, Sacha Krstulovic, Jon Barker, and Mark Plumbley',
            'name_remote': 'The CHiME-Home dataset is a collection of annotated domestic environment audio recordings',
            'url': None,
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': 'Unknown',
            'microphone_model': 'Unknown',
        }

        self.crossvalidation_folds = 5
        self.sample_mode = '.16kHz'

        self.package_list = [
            {
                'remote_package': 'https://archive.org/download/chime-home/chime_home.tar.gz',
                'local_package': os.path.join(self.local_path, 'chime_home.tar.gz'),
                'local_audio_path': os.path.join(self.local_path, 'chime_home', 'chunks'),
            },
        ]

    @property
    def audio_files(self):
        """Get all audio files in the dataset, use only files from CHime-Home-refined set.

        Parameters
        ----------
        nothing

        Returns
        -------
        files : list
            audio files

        """

        if self.files is None:
            self.files = []
            for item in self.package_list:
                path = item['local_audio_path']
                if path:
                    l = os.listdir(path)
                    for f in l:
                        file_name, file_extension = os.path.splitext(f)
                        if file_extension[1:] in self.audio_extensions and file_name.endswith(self.sample_mode):
                            if os.path.abspath(os.path.join(path, f)) not in self.files:
                                self.files.append(os.path.abspath(os.path.join(path, f)))
            self.files.sort()
        return self.files

    def read_chunk_meta(self, meta_filename):
        if os.path.isfile(meta_filename):
            meta_file_handle = open(meta_filename, 'rt')
            try:
                meta_file_reader = csv.reader(meta_file_handle, delimiter=',')
                data = {}
                for meta_file_row in meta_file_reader:
                    data[meta_file_row[0]] = meta_file_row[1]
            finally:
                meta_file_handle.close()
            return data

    def read_file_list(self, filename):
        files = []
        meta_data_handle = open(os.path.join(self.local_path, 'chime_home', filename))
        try:
            meta_file_reader = csv.reader(meta_data_handle, delimiter=',')
            for meta_file_row in meta_file_reader:
                files.append(meta_file_row[1])
        finally:
            meta_data_handle.close()
        return files

    def tagcode_to_taglabel(self, tag):
        map = {'c': 'child speech',
               'm': 'adult male speech',
               'f': 'adult female speech',
               'v': 'video game/tv',
               'p': 'percussive sound',
               'b': 'broadband noise',
               'o': 'other',
               'S': 'silence/background',
               'U': 'unidentifiable'
               }
        if tag in map:
            return map[tag]
        else:
            return None

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Legacy dataset meta files are converted to be compatible with current scheme.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            scene_label = 'home'

            files = self.read_file_list('development_chunks_refined.csv') + self.read_file_list('evaluation_chunks_refined.csv')
            meta = []
            for filename in self.audio_files:
                raw_path, raw_filename = os.path.split(filename)
                relative_path = self.absolute_to_relative(raw_path)

                base_filename, file_extension = os.path.splitext(raw_filename)
                annotation_filename = os.path.join(raw_path, base_filename.replace(self.sample_mode, '') + '.csv')
                meta_data = self.read_chunk_meta(annotation_filename)
                tags = []
                for i, tag in enumerate(meta_data['majorityvote']):
                    if tag is 'b':
                        print(filename)

                    if tag is not 'S' and tag is not 'U':
                        tags.append(self.tagcode_to_taglabel(tag))

                meta.append(
                    MetaDataItem({
                        'file': os.path.join(relative_path, raw_filename),
                        'scene_label': scene_label,
                        'tags': ';'.join(tags)+';',
                    })
                )
            self.meta_container.update(meta)
            self.meta_container.save()
        else:
            self.meta_container.load()

        all_folds_found = True
        for fold in xrange(1, self.crossvalidation_folds):
            train_filename = os.path.join(self.evaluation_setup_path,
                                          self._get_evaluation_setup_filename(setup_part='train', fold=fold))
            test_filename = os.path.join(self.evaluation_setup_path,
                                         self._get_evaluation_setup_filename(setup_part='test', fold=fold))

            if not os.path.isfile(train_filename):
                all_folds_found = False
            if not os.path.isfile(test_filename):
                all_folds_found = False

        if not all_folds_found:
            if not os.path.isdir(self.evaluation_setup_path):
                os.makedirs(self.evaluation_setup_path)

            dcase_crossval = {
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
            }
            with open(os.path.join(self.local_path, 'chime_home', 'development_chunks_refined_crossval_dcase2016.csv'), 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    dcase_crossval[int(row[2])+1].append(
                        self.relative_to_absolute_path(
                            os.path.join('chime_home', 'chunks', row[1] + self.sample_mode + '.wav')
                        )
                    )

            for fold in range(1, self.crossvalidation_folds+1):
                # Collect training and testing files
                train_files = []
                for f in range(1, self.crossvalidation_folds+1):
                    if f is not fold:
                        train_files += dcase_crossval[f]
                test_files = dcase_crossval[fold]

                # Create meta containers and save them

                # Train
                train_filename = os.path.join(self.evaluation_setup_path,
                                              self._get_evaluation_setup_filename(setup_part='train', fold=fold))

                train_meta = MetaDataContainer(filename=train_filename)
                for filename in train_files:
                    train_meta.append(self.file_meta(filename)[0])
                train_meta.save()

                # Test
                test_filename = os.path.join(self.evaluation_setup_path,
                                             self._get_evaluation_setup_filename(setup_part='test', fold=fold))
                test_meta = MetaDataContainer(filename=test_filename)
                for filename in test_files:
                    test_meta.append(MetaDataItem({'file': self.absolute_to_relative(filename)}))
                test_meta.save()

                # Evaluate
                eval_filename = os.path.join(self.evaluation_setup_path,
                                             self._get_evaluation_setup_filename(setup_part='evaluate', fold=fold))
                eval_meta = MetaDataContainer(filename=eval_filename)
                for filename in test_files:
                    eval_meta.append(self.file_meta(filename)[0])
                eval_meta.save()

# =====================================================
# Others
# =====================================================
class TUT_SED_Synthetic_2016(SoundEventDataset):
    """TUT SED Synthetic 2016

    """

    def __init__(self, *args, **kwargs):
        kwargs['storage_name'] = kwargs.get('storage_name', 'TUT-SED-synthetic-2016')
        super(TUT_SED_Synthetic_2016, self).__init__(*args, **kwargs)

        self.dataset_group = 'sound event'
        self.dataset_meta = {
            'authors': 'Emre Cakir',
            'name_remote': 'TUT-SED Synthetic 2016',
            'url': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/tut-sed-synthetic-2016',
            'audio_source': 'Field recording',
            'audio_type': 'Synthetic',
            'recording_device_model': 'Unknown',
            'microphone_model': 'Unknown',
        }

        self.crossvalidation_folds = 1

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.audio.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.audio.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.audio.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.audio.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/datasets/TUT-SED-synthetic-2016/TUT-SED-synthetic-2016.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-SED-synthetic-2016.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
        ]

    def _after_extract(self, to_return=None):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        if not self.meta_container.exists():
            meta_data = MetaDataContainer()
            for filename in self.audio_files:
                raw_path, raw_filename = os.path.split(filename)

                relative_path = self.absolute_to_relative(raw_path)
                base_filename, file_extension = os.path.splitext(raw_filename)
                annotation_filename = os.path.join(
                    self.local_path,
                    relative_path.replace('audio', 'meta'),
                    base_filename.replace('-mix-', '-annot-') + '.txt'
                )

                data = MetaDataContainer(filename=annotation_filename).load()
                for item in data:
                    item['file'] = os.path.join(relative_path, raw_filename)
                    item['scene_label'] = 'synthetic'
                    item['source_label'] = 'mixture'

                meta_data += data

            self.meta_container.update(meta_data)
            self.meta_container.save()
        else:
            self.meta_container.load()

    def _get_evaluation_setup_filename(self, setup_part='train', fold=None, scene_label=None, file_extension='txt'):
        parts = []
        if scene_label:
            parts.append(scene_label)

        if fold:
            parts.append('fold' + str(fold))

        if setup_part == 'train':
            return os.path.join(self.evaluation_setup_path, 'train+validate' + '.' + file_extension)
        elif setup_part == 'test':
            return os.path.join(self.evaluation_setup_path, 'test' + '.' + file_extension)
        elif setup_part == 'validate':
            return os.path.join(self.evaluation_setup_path, 'validate' + '.' + file_extension)
        elif setup_part == 'evaluate':
            return os.path.join(self.evaluation_setup_path, 'evaluate' + '.' + file_extension)

    def validation_files(self, fold=0, scene_label=None):
        validation_files = MetaDataContainer(
            filename=self._get_evaluation_setup_filename(setup_part='validate', fold=fold)
        ).load().file_list

        for index, filename in enumerate(validation_files):
            validation_files[index] = self.relative_to_absolute_path(filename)

        return validation_files

