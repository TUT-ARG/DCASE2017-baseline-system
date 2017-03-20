.. _usage_tutorial:
.. tabularcolumns:: |p{1.5cm}|p{1.5cm}|L|

Usage
=====

For each task there is a separate application (.py file) in ``applications/`` directory:

+------------------------------------------------+------------------------------------------------------------------------------------------+
| ``applications/task1.py``                      | DCASE2017 baseline for Task 1, Acoustic scene classification                             |
+------------------------------------------------+------------------------------------------------------------------------------------------+
| ``applications/task2.py``                      | DCASE2017 baseline for Task 2, Detection of rare sound events                            |
+------------------------------------------------+------------------------------------------------------------------------------------------+
| ``applications/task3.py``                      | DCASE2017 baseline for Task 3, Sound event detection in real life audio                  |
+------------------------------------------------+------------------------------------------------------------------------------------------+

Application arguments
---------------------

All the usage arguments are shown by ``python task1.py -h``.

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-h``, ``--help``                              | Application help.                                                                       |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-o``, ``--overwrite``                         | Force overwrite mode.                                                                   |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-v``, ``--version``                           | Show application version.                                                               |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

**System mode**

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-m {dev, challenge}``                         | System mode, ``dev`` for normal development, ``challenge`` for challenge type behaviour |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

Each application has two operating modes: **Development mode** and **Challenge mode**. In development mode, the *development dataset* is used with the cross-validation setup: training applied for training set, and testing for testing set. In challenge mode, the *development dataset* is fully used for training the acoustic models, and a second dataset, *evaluation dataset*, is used for testing to generate system outputs (if ground truth is available for the evaluation dataset, the output is also evaluated).
This mode is designed to be used when running the system on the evaluation dataset, for generating the system outputs for the challenge submission.


**System parameters**

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-p FILE``, ``--parameters FILE``              | Parameter file (YAML) to overwrite the default parameters                               |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-s PARAMETER_SET``,                           | Parameter set id. Can be also comma separated list e.g. ``-s set1,set2,set3``.          |
| ``--parameter_set PARAMETER_SET``               | In this case, each set is run separately.                                               |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

The application supports multi-level parameter overwriting, to enable flexible switching between different system setups.
The **default parameters** are defined in ``applications/parameters/task?.defaults.yaml``, and these parameters are replaced by **parameter set** for the current run.
Define here only parameters that you want to overwrite (compared to the defaults).

More about :ref:`parameterization <parameterization>`

**Information**

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-show_set``                                   | List of available parameter sets                                                        |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-show_datasets``                              | List of available datasets                                                              |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-show_parameters``                            | Show current parameters                                                                 |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-show_eval``                                  | Show evaluation results                                                                 |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

**System printing**

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``-n``, ``--node``                              | Node mode, console printing tuned for computer grid usage                               |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

Basic usage
-----------

With default settings, the system will download the needed datasets and extract them under directory ``data`` (storage path is controlled with parameter ``path->data``), and proceed to train and evaluate the system, for example::


    python task1.py

To run all provided system setups one after another::

    python task1.py -s dcase2017,dcase2017_gpu,gmm,minimal

For development with the system, one should create a new parameter set file in order to overwrite the default parameters with it::

    python task1.py -p custom.yaml -s custom_set

Example of ``custom.yaml`` file:

.. code-block:: yaml

    active_set: custom_set

    sets:
        - set_id: custom_set
          feature_extractor:
            win_length_seconds: 0.1
            hop_length_seconds: 0.5

To run the system in **challenge mode**::

    python task1.py -p custom.yaml -s custom_set -m challenge