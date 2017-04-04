.. _install:

Installation
============

**Download the system**

Download the `latest package <https://github.com/TUT-ARG/DCASE2017-baseline-system/archive/master.zip>`_ or clone
the system directly from the repository. To clone repository with HTTPS::

    git clone https://github.com/TUT-ARG/DCASE2017-baseline-system.git

or with ssh::

    git clone git@github.com:TUT-ARG/DCASE2017-baseline-system.git


The system is developed for `Python 2.7 <https://www.python.org/>`_, `Python 3.5 <https://www.python.org/>`_ and `Python 3.6 <https://www.python.org/>`_.
The system is tested to work on Linux, Windows and MacOS platforms.

One can install either the official `CPython <https://www.python.org/>`_ or use some Python distribution based on it. New users are recommended to use `Anaconda Python distribution <https://www.continuum.io/>`_.


**External modules**

To ensure that all external modules are installed, run command::

    pip install -r requirements.txt

**PySoundFile**

`PySoundFile <https://github.com/bastibe/PySoundFile>`_ is used to read and write audio files in the system. The library depends on system level library `libsndfile <http://www.mega-nerd.com/libsndfile/>`. Under Linux (Debian/Ubuntu) you can install this with command::

    sudo apt-get install libsndfile1


**Theano**

The system uses by default `Theano <http://deeplearning.net/software/theano/>`_ as `Keras <https://keras.io/>`_ backend.

There was a bug in Theano 0.8.2 version, so make sure to use 0.9.0 release. The bug will affect Task 2 system (when using GPU and binary_crossentropy as loss function). To fix this, make sure you have installed correct version of Theano::

    pip install theano==0.9.0

Or use latest from git with command::

    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git


**Tensorflow**

It is also possible to use `Tensorflow <https://www.tensorflow.org/>`_ as `Keras <https://keras.io/>`_ backend. If you plan to use it with GPU make sure you have installed GPU enabled package::

    pip uninstall tensorflow
    pip uninstall tensorflow-gpu
    pip install tensorflow-gpu

Current version of Tensorflow only supports CUDA 3.0 compatible graphic cards.

**Audio datasets**

The system will automatically download the needed audio datasets, and place them under the directory specified in the parameters (see parameter ``path->data``).

+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------+
| Dataset                                                                                                                          | Type                             | Audio files     | Size on disk    | License                                                                                      |
+==================================================================================================================================+==================================+=================+=================+==============================================================================================+
| `TUT Acoustic scenes 2017, development <https://zenodo.org/record/400516>`_                                                      | Acoustic scene                   | 4680            | 22Gb            | **Academic use only**                                                                        |
|                                                                                                                                  |                                  |                 |                 | (see EULA inside                                                                             |
|                                                                                                                                  |                                  |                 |                 | the package)                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------+
| `TUT Rare sound events 2017, development <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-rare-sound-event-detection>`_    | Sound events /                   | 1281            | 9.2Gb           | **Academic use only**                                                                        |
|                                                                                                                                  | synthetic                        |                 |                 | (see EULA inside                                                                             |
|                                                                                                                                  |                                  |                 |                 | the package)                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------+
| `TUT Sound events 2017, development <https://zenodo.org/record/400515>`_                                                         | Sound events                     | 24              | 2.6Gb           | **Academic use only**                                                                        |
|                                                                                                                                  | realistic                        |                 |                 | (see EULA inside                                                                             |
|                                                                                                                                  |                                  |                 |                 | the package)                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------+-----------------+----------------------------------------------------------------------------------------------+