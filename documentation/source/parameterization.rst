.. _parameterization:

Parameterization
================

The baseline system supports multi-level parameter overwriting, to enable flexible switching between different system setups.
Parameter changes are tracked with hashes calculated from parameter sections. These parameter hashes are used in the storage file paths when saving data (features, model, or results).
By using this approach, the system will compute features, models and results only once for the specific parameter set, and after that it will reuse this precomputed data.

Parameter overwriting
---------------------

Parameters are stored in YAML-formatted files, which are handled internally in the system as Dictionaries. **Default parameters** is the set of all possible parameters recognized by the system. These default parameters are defined in ``applications/parameters/task?.defaults.yaml``.
**Parameter set** is a smaller set of parameters used to overwrite values of the default parameters. This can be used to select methods for processing, or tune parameters.


Parameter file
--------------

Parameters files are YAML-formatted files, containing the following three blocks:

- ``active_set``, default parameter set id
- ``sets``, list of dictionaries
- ``defaults`` dictionary containing default parameters which are overwritten by the ``sets[active_set]``

At the top level of the parameter dictionary there are ``parameter sections``; depending on the name of the section, the parameters inside it are processed sometimes differently (See below more information.)

Example file:

.. code-block:: yaml

    active_set: SET1

    sets:
        - set_id: SET1
          flow:
            task1: false
            task2: false
            task3: true
          section1:
            enable: true
            field1: 11025
          section2
            enable: true
            field2: 44100
        - set_id: SET2
          section1:
            enable: false
          section2
            enable: false

    defaults:
        flow:
            task1: true
            task2: true
            task3: true

        section1:
            enable: false
            field1: 44100
            field2: 22050

        section2:
            enable: true
            field1: 44100
            field2: 22050

Parameter hash
--------------

Parameter hashes are MD5 hashes calculated for each parameter section. In order to make these hashes more robust, some pre-processing is applied before hash calculation:

- If section contains field ``enable`` with value ``False``, all fields inside this section are excluded from the parameter hash calculation. This will avoid recalculating the hash if the section is not used but some of these unused parameters are changed.
- If section contains fields with value ``False``, these fields are excluded from the parameter hash calculation. This will enable to add new flag parameters without changing the hash. Define the new flag such that the previous behaviour is happening when this field is set to false.
- All ``non_hashable_fields`` fields are excluded from the parameter hash calculation. These fields are set when ParameterContainer is constructed, and they usually are fields used to print various values to the console. These fields do not change the system output to be saved onto disk, and hence they are excluded from hash.

Parameter sections
------------------

The functionality of the parameters depending on the section name.

Flow
^^^^

The processing blocks of the system can be controlled through this section. Usually all of them can be kept on.

Example section:

.. code-block:: yaml

    flow:
        initialize: true
        extract_features: true
        feature_normalizer: true
        train_system: true
        test_system: true
        evaluate_system: true

+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| extract_features               | bool         | Initialize the system                                                |
+--------------------------------+--------------+----------------------------------------------------------------------+
| feature_normalizer             | bool         | Extract acoustic features for all data at once.                      |
+--------------------------------+--------------+----------------------------------------------------------------------+
| train_system                   | bool         | Train the system with training material                              |
+--------------------------------+--------------+----------------------------------------------------------------------+
| test_system                    | bool         | Test the system with testing material                                |
+--------------------------------+--------------+----------------------------------------------------------------------+
| evaluate_system                | bool         | Evaluate correctness of the system outputs produced in the           |
|                                |              | ``test_system`` block.                                               |
+--------------------------------+--------------+----------------------------------------------------------------------+


General
^^^^^^^

This section contains general settings, mostly related to printing and logging.

Example section:

.. code-block:: yaml

    general:
        overwrite: false

        challenge_submission_mode: false

        print_system_progress: true
        log_system_parameters: false
        log_system_progress: false

+--------------------------------+-----------------------+----------------------------------------------------------------------+
| Field name                     | Value type            | Description                                                          |
+================================+=======================+======================================================================+
| overwrite                      | bool                  | Overwrite all pre-calculated data.                                   |
|                                |                       | Enable this when changing system implementation.                     |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| challenge_submission_mode      | bool                  | Save results to path location defined in ``path->challenge_results``.|
|                                |                       | Use this mode when preparing a submission to the challenge.          |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| print_system_progress          | bool                  | Print the system progress into console using carriage return.        |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| use_ascii_progress_bar         | bool                  | Force ASCII progres bars, use this if your console does not support  |
|                                |                       | UTF-8 character set. Under Windows this is set automatically True.   |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| log_system_parameters          | bool                  | Save system parameters into system log file.                         |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| log_system_progress            | bool                  | Save system progress into system log file.                           |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| scene_handling                 | string                | Scene handling type, can be used in sound event detection            |
|                                | {scene-dependent |    | application to control how audio material from                       |
|                                |  scene-independent}   | multiple acoustic scene classes are handled.                         |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| active_scenes                  | list                  | List of active scene classes in the processing. This can be used     |
|                                |                       | to speed up processing when debugging.                               |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| event_handling                 | string                | Event handling type, can be used in binary sound event detection     |
|                                | {event-dependent |    | application to control how audio material from                       |
|                                |  event-independent}   | multiple event classes are handled.                                  |
+--------------------------------+-----------------------+----------------------------------------------------------------------+
| active_events                  | list                  | List of active event classes in the processing. This can be used     |
|                                |                       | to speed up processing when debugging.                               |
+--------------------------------+-----------------------+----------------------------------------------------------------------+

Path
^^^^

This section defines all paths for the system. Paths can be defined either as absolute or relative to the application code file.
Relative paths are converted into absolute before they are used.

Example section:

.. code-block:: yaml

    path:
        data: data/

        system_base: system/task1/
        feature_extractor: feature_extractor/
        feature_normalizer: feature_normalizer/
        learner: learner/
        recognizer: recognizer/
        evaluator: evaluator/

        recognizer_challenge_output: challenge_submission/task1/
        logs: logs/


+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| data                           | string       | Path to store all audio datasets.                                    |
+--------------------------------+--------------+----------------------------------------------------------------------+
| system_base                    | string       | Base path for the system to store all data.                          |
+--------------------------------+--------------+----------------------------------------------------------------------+
| feature_extractor              | string       | Directory name under system_base for extracted features              |
+--------------------------------+--------------+----------------------------------------------------------------------+
| feature_normalizer             | string       | Directory name under system_base for feature normalization values    |
+--------------------------------+--------------+----------------------------------------------------------------------+
| learner                        | string       | Directory name under system_base for learned acoustic models         |
+--------------------------------+--------------+----------------------------------------------------------------------+
| recognizer                     | string       | Directory name under system_base for predicted system outputs        |
+--------------------------------+--------------+----------------------------------------------------------------------+
| evaluator                      | string       | Directory name under system_base for evaluated metric values         |
+--------------------------------+--------------+----------------------------------------------------------------------+
| recognizer_challenge_output    | string       | Path to store system output in challenge mode.                       |
+--------------------------------+--------------+----------------------------------------------------------------------+
| logs                           | string       | Path to save system logs.                                            |
+--------------------------------+--------------+----------------------------------------------------------------------+


Dataset
^^^^^^^

This section defines the dataset use in **development mode** and in **challenge mode**.

Example section:

.. code-block:: yaml

    dataset:
        method: development

    dataset_method_parameters:
        development:
            name: TUTAcousticScenes_2017_DevelopmentSet
            fold_list: [1, 2, 3, 4]
            evaluation_mode: folds

        challenge_train:
            name: TUTAcousticScenes_2017_DevelopmentSet
            evaluation_mode: full

        challenge_test:
            name: TUTAcousticScenes_2017_EvaluationSet
            evaluation_mode: full

``dataset->method`` is used to select the active dataset.


+----------------------------------------------------+--------------+----------------------------------------------------------------------+
| Field name                                         | Value type   | Description                                                          |
+====================================================+==============+======================================================================+
| dataset->method                                    | string       | Active dataset, used to select parameter set                         |
|                                                    |              | from dataset_method_parameters                                       |
+----------------------------------------------------+--------------+----------------------------------------------------------------------+
| dataset_method_parameters->method->name            | string       | Dataset class name, use ``./task1.py -show_datasets``                |
|                                                    |              | to see valid ones                                                    |
+----------------------------------------------------+--------------+----------------------------------------------------------------------+
| dataset_method_parameters->method->fold_list       | list of ints | List of active folds. If nothing set, all available folds are used.  |
|                                                    |              | Use this to run the system on a subset of cross-validation folds.    |
+----------------------------------------------------+--------------+----------------------------------------------------------------------+
| dataset_method_parameters->method->evaluation_mode | string       | System evalution mode. With ``folds``, cross-evaluation folds are    |
|                                                    | {full|folds} | used. With ``full`` all the data is used for training and testing.   |
+----------------------------------------------------+--------------+----------------------------------------------------------------------+


Feature extractor
^^^^^^^^^^^^^^^^^

This section defines the general feature extraction parameters and extractor specific parameters.
``feature_stacker->stacking_recipe`` is used to select active feature extractors.

Example section:

.. code-block:: yaml

    feature_extractor:
        fs: 44100                               # Sampling frequency
        win_length_seconds: 0.04                # Window length
        hop_length_seconds: 0.02                # Hop length

+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| fs                             | int          | Sampling frequency. If different sampling frequency is encountered   |
|                                |              | during audio file loading, resampling is used.                       |
+--------------------------------+--------------+----------------------------------------------------------------------+
| win_length_seconds             | float        | Analysis window length in seconds.                                   |
+--------------------------------+--------------+----------------------------------------------------------------------+
| hop_length_seconds             | float        | Analysis window hop length in seconds.                               |
+--------------------------------+--------------+----------------------------------------------------------------------+

Example section:

.. code-block:: yaml

    feature_extractor_method_parameters:
        mel:                                    # Mel band energy
            mono: true                          # [true, false]
            window: hamming_asymmetric          # [hann_asymmetric, hamming_asymmetric]
            spectrogram_type: magnitude         # [magnitude, power]
            n_mels: 40                          # Number of MEL bands used
            normalize_mel_bands: false          # [true, false]
            n_fft: 2048                         # FFT length
            fmin: 0                             # Minimum frequency when constructing MEL bands
            fmax: 22050                         # Maximum frequency when constructing MEL band
            htk: false                          # Switch for HTK-styled MEL-frequency equation
            log: true                           # Logarithmic

        mfcc:                                   # Mel-frequency cepstral coefficients
            mono: true                          # [true, false]
            window: hamming_asymmetric          # [hann_asymmetric, hamming_asymmetric]
            spectrogram_type: magnitude         # [magnitude, power]
            n_mfcc: 20                          # Number of MFCC coefficients
            n_mels: 40                          # Number of MEL bands used
            n_fft: 2048                         # FFT length
            fmin: 0                             # Minimum frequency when constructing MEL bands
            fmax: 22050                         # Maximum frequency when constructing MEL band
            htk: false                          # Switch for HTK-styled MEL-frequency equation

        mfcc_delta:                             # MFCC delta coefficients
            width: 9                            #

        mfcc_acceleration:                      # MFCC acceleration coefficients
            width: 9                            #

+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| Field name                     | Value type                             | Description                                                          |
+================================+========================================+======================================================================+
| **feature_extractor_method_parameters->mel**                                                                                                   |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| mono                           | bool                                   | If true, multi-channel audio input is averaged into single channel.  |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| window                         | string                                 | Analysis window function.                                            |
|                                | {hann_asymmetric | hamming_asymmetric} |                                                                      |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| spectrogram_type               | string                                 | Spectrogram type.                                                    |
|                                | {magnitude | power}                    |                                                                      |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| n_mels                         | int                                    | Number of mel bands used.                                            |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| normalize_mel_bands            | bool                                   | Normalize mel bands.                                                 |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| n_fft                          | int                                    | FFT length.                                                          |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| fmin                           | int                                    | Minimum frequency when constructing mel bands                        |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| fmax                           | int                                    | Maximum frequency when constructing mel band                         |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| htk                            | bool                                   | Switch for HTK-style mel-frequency equation                          |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| log                            | bool                                   | Logarithmic                                                          |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| **feature_extractor_method_parameters->mfcc**                                                                                                  |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| mono                           | bool                                   | If true, multi-channel audio input is averaged into single channel.  |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| window                         | string                                 | Analysis window function.                                            |
|                                | {hann_asymmetric | hamming_asymmetric} |                                                                      |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| spectrogram_type               | string                                 | Spectrogram type.                                                    |
|                                | {magnitude | power}                    |                                                                      |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| n_mfcc                         | int                                    | Number of mfcc coefficients. Zeroth coefficient is always returned.  |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| n_mels                         | int                                    | Number of mel bands used.                                            |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| n_fft                          | int                                    | FFT length.                                                          |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| fmin                           | int                                    | Minimum frequency when constructing mel bands                        |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| fmax                           | int                                    | Maximum frequency when constructing mel band                         |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| htk                            | bool                                   | Switch for HTK-style mel-frequency equation                          |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| **feature_extractor_method_parameters->mfcc_delta**                                                                                            |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| width                          | int                                    | Delta window length.                                                 |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| **feature_extractor_method_parameters->mfcc_acceleration**                                                                                     |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+
| width                          | int                                    | Delta-delta window length.                                           |
+--------------------------------+----------------------------------------+----------------------------------------------------------------------+

Feature stacker
^^^^^^^^^^^^^^^

This section defines how the extracted features are combined to form the feature vector (and feature matrix).
Stacking recipe is ``;`` limited string with stacking recipe item in specific format:

- ``[extractor (string)]``, default channel 0 and full vector
- ``[extractor (string)]=[start index (int)]-[end index (int)]``, default channel 0 and vector [start:end]
- ``[extractor (string)]=[channel (int)]:[start index (int)]-[end index (int)]``, specified channel and vector [start:end]
- ``[extractor (string)]=1,2,3,4,5``, default channel 0 and vector [1,2,3,4,5]
- ``[extractor (string)]=0``, specified channel and full vector

For example to get feature vector with mfcc and omitting zeroth coefficient use::

    stacking_recipe: mfcc=1-19;mfcc_delta;mfcc_acceleration

For example to get features from both channels (make sure ``feature_extractor_method_parameters->mel->mono`` field is set to false::

    stacking_recipe: mel=0;mel=1


Example section:

.. code-block:: yaml

    feature_stacker:
        stacking_recipe: mel

+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| stacking_recipe                | string       | Stacking recipe to form feature vector.                              |
+--------------------------------+--------------+----------------------------------------------------------------------+
| feature_hop                    | int          | Debugging parameter to strip data by taking every Nth feature vector.|
|                                | {1}          | Use this only for classification tasks, as it will break             |
|                                |              | synchronization of the meta data.                                    |
+--------------------------------+--------------+----------------------------------------------------------------------+

Feature normalizer
^^^^^^^^^^^^^^^^^^

This section defines the feature normalization.

Example section:

.. code-block:: yaml

    feature_normalizer:
        enable: true
        type: global

+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| enable                         | bool         | Switch to enable feature normalization.                              |
+--------------------------------+--------------+----------------------------------------------------------------------+
| type                           | string       | Normalization type. Currently only global normalization              |
|                                | {global}     | supported.                                                           |
+--------------------------------+--------------+----------------------------------------------------------------------+

Feature aggregator
^^^^^^^^^^^^^^^^^^

This section defines the feature aggregation.
The feature aggregator can be used to process the feature matrix inside the processing window. It can be used for example
to collapse features within the window by calculating mean and std per feature item, or to flatten the matrix into single longer feature vector.

Supported processing methods:

- ``flatten``
- ``mean``
- ``std``
- ``cov``
- ``kurtosis``
- ``skew``

The processing methods can combined with ``;``.

For example, to calculate mean and std::

    aggregation_recipe: mean;std


Example section:

.. code-block:: yaml

    feature_aggregator:
        enable: false
        aggregation_recipe: flatten
        win_length_seconds: 0.1
        hop_length_seconds: 0.02


+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| enable                         | bool         | Switch to enable feature aggregation.                                |
+--------------------------------+--------------+----------------------------------------------------------------------+
| aggregation_recipe             | string       | Aggregation recipe. See formatting above.                            |
+--------------------------------+--------------+----------------------------------------------------------------------+
| win_length_seconds             | float        | Aggregation processing window length.                                |
+--------------------------------+--------------+----------------------------------------------------------------------+
| hop_length_seconds             | float        | Aggregation processing window hop length.                            |
+--------------------------------+--------------+----------------------------------------------------------------------+

Learner
^^^^^^^

This section defines the learner stage of the system.

Example section:

.. code-block:: yaml

    learner:
        method: mlp

        audio_error_handling: false
        show_model_information: false


+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| method                         | string       | Learner method name. Used to select parameters                       |
|                                |              | from ``learner_method_parameters``.                                  |
+--------------------------------+--------------+----------------------------------------------------------------------+
| audio_error_handling           | bool         | Switch to skip frames annotated to contain errors.                   |
|                                |              | Only used in Task1 application                                       |
+--------------------------------+--------------+----------------------------------------------------------------------+
| show_model_information         | bool         | Switch to show extra information about the learned model. Used       |
|                                |              | only with keras learners.                                            |
+--------------------------------+--------------+----------------------------------------------------------------------+
| file_hop                       | int          | Debugging parameter to strip data by taking every Nth file when      |
|                                | {1}          | collecting training data. Use this for debugging when dealing with   |
|                                |              | large datasets.                                                      |
+--------------------------------+--------------+----------------------------------------------------------------------+

**MLP**

Example section for MLP based learner:

.. code-block:: yaml

    learner_method_parameters:
        mlp:
            seed: 1

            keras:
                backend: theano
                backend_parameters:
                    floatX: float32
                    device: cpu
                    fastmath: false

            validation:
                enable: true
                setup_source: generated_scene_balanced
                validation_amount: 0.10
                seed: 1

            training:
                nb_epoch: 100
                batch_size: 256
                shuffle: true
                callbacks:
                    - type: EarlyStopping
                      parameters:
                          monitor: val_categorical_accuracy
                          min_delta: 0.001
                          patience: 10
                          verbose: 0
                          mode: max
            model:
                config:
                    - class_name: Dense
                      config:
                        units: 50
                        kernel_initializer: uniform
                        activation: relu

                    - class_name: Dropout
                      config:
                        rate: 0.2

                    - class_name: Dense
                      config:
                        units: 50
                        kernel_initializer: uniform
                        activation: relu

                    - class_name: Dropout
                      config:
                        rate: 0.2

                    - class_name: Dense
                      config:
                        units: CLASS_COUNT
                        kernel_initializer: uniform
                        activation: softmax

                loss: categorical_crossentropy

                optimizer:
                    type: Adam

                metrics:
                    - categorical_accuracy

This learner is using Keras neural network implementation. See `documentation <https://keras.io/>`_.

+--------------------------------+--------------+------------------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                                  |
+================================+==============+==============================================================================+
| seed                           | int          | Randomization seed. Use this to make learner behaviour                       |
|                                |              | deterministic.                                                               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->keras**                                                                                                               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| backend                        | string       | Keras backend selector.                                                      |
|                                | {theano |    |                                                                              |
|                                | tensorflow}  |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->keras->backend_parameters**                                                                                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| device                         | string       | Device selector. ``cpu`` is best option to produce deterministic             |
|                                | {cpu | gpu}  | results. All baseline results are calculated in cpu mode.                    |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| floatX                         | string       | Float number type. Usually float32 used since that is compatible             |
|                                |              | with GPUs. Valid only for ``theano`` backend.                                |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| fastmath                       | bool         | If true, will enable fastmath mode when CUDA code is compiled.               |
|                                |              | Div and sqrt are faster, but precision is lower. This can cause              |
|                                |              | numerical issues some in cases. Valid only for ``theano`` backend            |
|                                |              | and GPU mode.                                                                |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| optimizer                      | string       | Compilation mode for theano functions.                                       |
|                                | {fast_run |  |                                                                              |
|                                | merge |      |                                                                              |
|                                | fast_compile |                                                                              |
|                                | None}        |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| openmp                         | bool         | If true, Theano will use multiple cores, see                                 |
|                                |              | `more <http://deeplearning.net/software/theano/tutorial/multi_cores.html>`_. |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| threads                        | int          | Number of threads used. Use one to disable threading.                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| CNR                            | bool         | Conditional numerical reproducibility for MKL BLAS. When set to True,        |
|                                |              | compatible mode used.                                                        |
|                                |              | See `more <https://software.intel.com/en-us/node/528408>`_.                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->validation**                                                                                                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, validation set is used during the training procedure.               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| setup_source                   | string       | Validation setup source. Valid sources:                                      |
|                                |              |                                                                              |
|                                |              | - ``generated_scene_balanced``, balanced based on scene labels,              |
|                                |              |   used for Task1.                                                            |
|                                |              | - ``generated_event_file_balanced``, balanced based on events, used          |
|                                |              |   for Task2.                                                                 |
|                                |              | - ``generated_scene_location_event_balanced``, balanced based on             |
|                                |              |   scene, location and events. Used for Task3.                                |
|                                |              | - ``dataset``, validation set specified by dataset in use.                   |
|                                |              |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| validation_amount              | float        | Percentage of training data selected for validation. Use value               |
|                                |              | between 0.0-1.0. Valid only if validation setup is generated.                |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| seed                           | int          | Validation set generation seed. If Null, learner seed will be used.          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->training**                                                                                                            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| epochs                         | int          | Number of epochs.                                                            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| batch_size                     | int          | Batch size.                                                                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| shuffle                        | bool         | If true, training samples are shuffled at each epoch.                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->training->callbacks**, list of parameter sets in following format. Callback called during the model training.         |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| type                           | string       | Callback name, use standard keras callbacks                                  |
|                                |              | `callbacks <https://keras.io/callbacks/>`_ or ones defined by                |
|                                |              | dcase_framework (Plotter, Stopper, Stasher).                                 |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| parameters                     | dict         | Place inside this all parameters for the callback.                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->training->model->config**, list of dicts. Defining network topology.                                                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| class_name                     | string       | Layer name. Use standard keras                                               |
|                                |              | `core layers <https://keras.io/layers/core/>`_,                              |
|                                |              | `convolutional layers <https://keras.io/layers/convolutional/>`_,            |
|                                |              | `pooling layers <https://keras.io/layers/pooling/>`_,                        |
|                                |              | `recurrent layers <https://keras.io/layers/recurrent/>`_, or                 |
|                                |              | `normalization layers <https://keras.io/layers/normalization/>`_.            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| config                         | dict         | Place inside this all parameters for the layer.                              |
|                                |              | See Keras documentation. Magic parameter values:                             |
|                                |              |                                                                              |
|                                |              | - ``FEATURE_VECTOR_LENGTH``, feature vector length.                          |
|                                |              |   This automatically inserted for input layer.                               |
|                                |              | - ``CLASS_COUNT``, number of classes.                                        |
|                                |              |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| input_shape                    | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->training->model**                                                                                                     |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| loss                           | string       | Keras loss function name. See                                                |
|                                |              | `Keras documentation <https://keras.io/losses/>`_.                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| metrics                        | list of      | Keras metric function name. See                                              |
|                                | strings      | `Keras documentation <https://keras.io/metrics/>`_.                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **mlp->training->model->optimizer**                                                                                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| type                           | string       | Keras optimizer name. See                                                    |
|                                |              | `Keras documentation <https://keras.io/optimizers/>`_.                       |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| parameters                     | dict         | Place inside this all parameters for the optimizer.                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+

**Keras sequential**

Example section for Keras sequential learner:

.. code-block:: yaml

    learner_method_parameters:
      keras_seq:
        seed: 0
        keras:
          backend: theano
          backend_parameters:
            floatX: float32
            device: gpu
            fastmath: true
            optimizer: fast_run
            openmp: true
            threads: 4
            CNR: true

        input_sequencer:
          enable: false

        temporal_shifter:
          enable: false

        generator:
          enable: false
          method: feature_generator
          max_q_size: 1
          workers: 1
          parameters:
            buffer_size: 10

        validation:
          enable: true
          setup_source: generated_event_file_balanced
          validation_amount: 0.10
          seed: 123

        training:
          epochs: 200
          batch_size: 256
          shuffle: true

          epoch_processing:
            enable: true

            external_metrics:
              enable: true
              evaluation_interval: 1
              metrics:
                - name: sed_eval.event_based.overall.error_rate.error_rate
                  label: ER
                  parameters:
                    evaluate_onset: true
                    evaluate_offset: false
                    t_collar: 0.5
                    percentage_of_length: 0.5
                - name: sed_eval.event_based.overall.f_measure.f_measure
                  label: F1
                  parameters:
                    evaluate_onset: true
                    evaluate_offset: false
                    t_collar: 0.5
                    percentage_of_length: 0.5

          callbacks:
            - type: Plotter
              parameters:
                interactive: true
                save: false
                output_format: pdf
                focus_span: 10
                plotting_rate: 5

            - type: Stopper
              parameters:
                monitor: sed_eval.event_based.overall.error_rate.error_rate
                initial_delay: 20
                min_delta: 0.01
                patience: 10

            - type: Stasher
              parameters:
                monitor: sed_eval.event_based.overall.error_rate.error_rate
                initial_delay: 20

        model:
          constants:
            LAYER_SIZE: 50
            LAYER_INIT: uniform
            LAYER_ACTIVATION: relu
            DROPOUT: 0.2

          config:
            - class_name: Dense
              config:
                units: LAYER_SIZE
                kernel_initializer: LAYER_INIT
                activation: LAYER_ACTIVATION

            - class_name: Dropout
              config:
                rate: DROPOUT

            - class_name: Dense
              config:
                units: LAYER_SIZE
                kernel_initializer: LAYER_INIT
                activation: LAYER_ACTIVATION

            - class_name: Dropout
              config:
                rate: DROPOUT

            - class_name: Dense
              config:
                units: CLASS_COUNT
                kernel_initializer: LAYER_INIT
                activation: sigmoid

          loss: binary_crossentropy

          optimizer:
            type: Adam

          metrics:
            - binary_accuracy

This learner is using Keras neural network implementation. See `documentation <https://keras.io/>`_.

+--------------------------------+--------------+------------------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                                  |
+================================+==============+==============================================================================+
| seed                           | int          | Randomization seed. Use this to make learner behaviour                       |
|                                |              | deterministic.                                                               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->keras**                                                                                                         |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| backend                        | string       | Keras backend selector.                                                      |
|                                | {theano |    |                                                                              |
|                                | tensorflow}  |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->keras->backend_parameters**                                                                                     |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| device                         | string       | Device selector. ``cpu`` is best option to produce deterministic             |
|                                | {cpu | gpu}  | results. All baseline results are calculated in cpu mode.                    |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| floatX                         | string       | Float number type. Usually float32 used since that is compatible             |
|                                |              | with GPUs. Valid only for ``theano`` backend.                                |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| fastmath                       | bool         | If true, will enable fastmath mode when CUDA code is compiled.               |
|                                |              | Div and sqrt are faster, but precision is lower. This can cause              |
|                                |              | numerical issues some in cases. Valid only for ``theano`` backend            |
|                                |              | and GPU mode.                                                                |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| optimizer                      | string       | Compilation mode for theano functions.                                       |
|                                | {fast_run |  |                                                                              |
|                                | merge |      |                                                                              |
|                                | fast_compile |                                                                              |
|                                | None}        |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| openmp                         | bool         | If true, Theano will use multiple cores, see                                 |
|                                |              | `more <http://deeplearning.net/software/theano/tutorial/multi_cores.html>`_. |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| threads                        | int          | Number of threads used. Use one to disable threading.                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| CNR                            | bool         | Conditional numerical reproducibility for MKL BLAS. When set to True,        |
|                                |              | compatible mode used.                                                        |
|                                |              | See `more <https://software.intel.com/en-us/node/528408>`_.                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->input_sequencer**, transforming input data into sequences                                                       |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, input sequencing is used during the training procedure.             |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| frames                         | int          | Frames per sequence                                                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| hop                            | int          | Hop (in frames) between sequences                                            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| padding                        | bool         | Replicating data when sequence is not full                                   |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->temporal_shifter**, shift data on temporal axis for each epoch                                                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, temporal data shifting per epoch is applied during the training     |
|                                |              | procedure.                                                                   |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| border                         | string       | How border is handled:                                                       |
|                                |              |                                                                              |
|                                | {roll |      | - ``roll``, data matrix is rolled (data moved from end to the begin)         |
|                                | push }       | - ``push``, unused material is not used.                                     |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| step                           | int          | How much sequence start is shifted per epoch                                 |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| max                            | int          | Maximum shift, after which shift is returned to zero.                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->generator**, data generator to read training data directly from disk during the training procedure.             |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, data generator is used to provide training material.                |
|                                |              |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| method                         | string       | Generator method:                                                            |
|                                | {feature}    | - feature, feature based generator                                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| max_q_size                     | int          | Maximum size for the generator queue                                         |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| workers                        | int          | Maximum number of generator processes to start up.                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->generator->parameters**                                                                                         |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| buffer_size                    | int          | Size of internal buffer. How many items (files) will be stored in the        |
|                                |              | memory.                                                                      |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->validation**                                                                                                    |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, validation set is used during the training procedure.               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| setup_source                   | string       | Validation setup source. Valid sources:                                      |
|                                |              |                                                                              |
|                                |              | - ``generated_scene_balanced``, balanced based on scene labels,              |
|                                |              |   used for Task1.                                                            |
|                                |              | - ``generated_event_file_balanced``, balanced based on events, used          |
|                                |              |   for Task2.                                                                 |
|                                |              | - ``generated_scene_location_event_balanced``, balanced based on             |
|                                |              |   scene, location and events. Used for Task3.                                |
|                                |              |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| validation_amount              | float        | Percentage of training data selected for validation. Use value               |
|                                |              | between 0.0-1.0.                                                             |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| seed                           | int          | Validation set generation seed. If Null, learner seed will be used.          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training**                                                                                                      |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| epochs                         | int          | Number of epochs.                                                            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| batch_size                     | int          | Batch size.                                                                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| shuffle                        | bool         | If true, training samples are shuffled at each epoch.                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->epoch_processing**, epoch by epoch processing outside Keras.                                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, training is done in smaller segments to allow evaluation of         |
|                                |              | external metrics for validation data.                                        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->epoch_processing->external_metrics**                                                                  |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| enable                         | bool         | If true, external metrics are evaluated.                                     |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| evaluation_interval            | int          | Evaluation is done every Nth epoch.                                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->epoch_processing->external_metrics->metrics**, list of dicts. Defining external metrics.              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| label                          | string       | Metric label, use unique label.                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| evaluator                      | string       | Evaluaor names:                                                              |
|                                |              |                                                                              |
|                                |              | - ``sed_eval.scene``, acoustic scene classification metrics                  |
|                                |              | - ``sed_eval.segment_based``, segment based sound event detection metrics    |
|                                |              | - ``sed_eval.event_based``, event based sound event detection metrics        |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| name                           | string       | Metric name, dict path to fetch metric value:                                |
|                                |              |                                                                              |
|                                |              | - ``overall.accuracy``, accuracy                                             |
|                                |              | - ``overall.f_measure.f_measure``, F1                                        |
|                                |              | - ``overall.error_rate.error_rate``, ER                                      |
|                                |              |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| parameters                     | dict         | Parameters for the evaluator. See                                            |
|                                |              | `sed_eval documentation <http://tut-arg.github.io/sed_eval/>`_.              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->callbacks**, list of parameter sets in following format. Callback called during the model training.   |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| type                           | string       | Callback name, use standard keras callbacks                                  |
|                                |              | `callbacks <https://keras.io/callbacks/>`_ or ones defined by                |
|                                |              | dcase_framework (Plotter, Stopper, Stasher).                                 |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| parameters                     | dict         | Place inside this all parameters for the callback.                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->model->constants**, Defined constant to be used in while defining network topology.                   |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->model->config**, list of dicts. Defining network topology.                                            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| class_name                     | string       | Layer name. Use standard keras                                               |
|                                |              | `core layers <https://keras.io/layers/core/>`_,                              |
|                                |              | `convolutional layers <https://keras.io/layers/convolutional/>`_,            |
|                                |              | `pooling layers <https://keras.io/layers/pooling/>`_,                        |
|                                |              | `recurrent layers <https://keras.io/layers/recurrent/>`_, or                 |
|                                |              | `normalization layers <https://keras.io/layers/normalization/>`_.            |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| config                         | dict         | Place inside this all parameters for the layer.                              |
|                                |              | See Keras documentation. Magic parameter values:                             |
|                                |              |                                                                              |
|                                |              | - ``FEATURE_VECTOR_LENGTH``, feature vector length.                          |
|                                |              |   This automatically inserted for input layer.                               |
|                                |              | - ``CLASS_COUNT``, number of classes.                                        |
|                                |              | - ``INPUT_SEQUENCE_LENGTH``, input sequence length                           |
|                                |              |                                                                              |
|                                |              | Addition constants defined in keras_seq->training->model->constants          |
|                                |              | can be used.                                                                 |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| input_shape                    | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| kernel_size                    | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| pool_size                      | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| dims                           | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| target_shape                   | list of      | List of integers which is converted into tuple before giving to Keras layer. |
|                                | ints         |                                                                              |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->model**                                                                                               |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| loss                           | string       | Keras loss function name. See                                                |
|                                |              | `Keras documentation <https://keras.io/losses/>`_.                           |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| metrics                        | list of      | Keras metric function name. See                                              |
|                                | strings      | `Keras documentation <https://keras.io/metrics/>`_.                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| **keras_seq->training->model->optimizer**                                                                                    |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| type                           | string       | Keras optimizer name. See                                                    |
|                                |              | `Keras documentation <https://keras.io/optimizers/>`_.                       |
+--------------------------------+--------------+------------------------------------------------------------------------------+
| parameters                     | dict         | Place inside this all parameters for the optimizer.                          |
+--------------------------------+--------------+------------------------------------------------------------------------------+


**GMM**

Example section for GMM based learner:

.. code-block:: yaml

    learner_method_parameters:
        gmm:
            n_components: 1
            covariance_type: diag
            tol: 0.001
            reg_covar: 0
            max_iter: 40
            n_init: 1
            init_params: kmeans
            random_state: 0


This learner is using ``sklearn.mixture.GaussianMixture`` implementation. See `documentation <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html/>`_.

+--------------------------------+--------------+----------------------------------------------------------------------+
| Field name                     | Value type   | Description                                                          |
+================================+==============+======================================================================+
| n_components                   | int          | The number of mixture components.                                    |
+--------------------------------+--------------+----------------------------------------------------------------------+
| covariance_type                | string       | Covariance type.                                                     |
|                                | { full |     |                                                                      |
|                                | tied |       |                                                                      |
|                                | diag |       |                                                                      |
|                                | spherical }  |                                                                      |
+--------------------------------+--------------+----------------------------------------------------------------------+
| tol                            | float        | Covariance threshold.                                                |
+--------------------------------+--------------+----------------------------------------------------------------------+
| reg_covar                      | float        | Non-negative regularization added to the diagonal of covariance.     |
+--------------------------------+--------------+----------------------------------------------------------------------+
| max_iter                       | int          | The number of EM iterations.                                         |
+--------------------------------+--------------+----------------------------------------------------------------------+
| n_init                         | int          | The number of initializations.                                       |
+--------------------------------+--------------+----------------------------------------------------------------------+
| init_params                    | string       | The method used to initialize model weights.                         |
|                                | { kmeans |   |                                                                      |
|                                | random }     |                                                                      |
+--------------------------------+--------------+----------------------------------------------------------------------+
| random_state                   | int          | Random seed.                                                         |
+--------------------------------+--------------+----------------------------------------------------------------------+

Recognizer
^^^^^^^^^^

This section defines the recognizer stage of the system.

Example section for Task 1:

.. code-block:: yaml

    recognizer:
        enable: true
        audio_error_handling: false

        frame_accumulation:
            enable: false
            type: sum

        frame_binarization:
            enable: false
            type: frame_max
            threshold: null

        decision_making:
            enable: true
            type: majority_vote

+--------------------------------+--------------------+----------------------------------------------------------------------+
| Field name                     | Value type         | Description                                                          |
+================================+====================+======================================================================+
| enable                         | bool               | Section selector                                                     |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| audio_error_handling           | bool               | Switch to skip frames annotated to contain errors.                   |
|                                |                    | Only used in Task1 application. This used to exclude temporary       |
|                                |                    | microphone failure and radio signal interferences from mobile phones.|
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **frame_accumulation**, Defining frame probability accumulation.                                                           |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable frame probability accumulation.                               |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Operator type used to accumulate.                                    |
|                                | {sum |             |                                                                      |
|                                | prod |             |                                                                      |
|                                | mean }             |                                                                      |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **frame_binarization**, Defining frame probability binarization.                                                           |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable frame probability binarization.                               |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Type of binarization:                                                |
|                                | {frame_max |       |                                                                      |
|                                | global_threshold } | - ``frame_max``, each frame is treated individually, max of each     |
|                                |                    |   frame is set to one, other zero.                                   |
|                                |                    | - ``global_threshold``, global threshold, all values over the        |
|                                |                    |   threshold are set to one.                                          |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| threshold                      | float              | Threshold value. Set to null if not used.                            |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **decision_making**, Defining final decision making.                                                                       |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable final decision making.                                        |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Type of decision:                                                    |
|                                | {maximum |         |                                                                      |
|                                | majority_vote }    | - ``maximum``, maximum probability is choosen.                       |
|                                |                    | - ``majority_vote``, majority vote among binarized frame decisions.  |
+--------------------------------+--------------------+----------------------------------------------------------------------+

Example section for Task 2 and Task 3:

.. code-block:: yaml

    recognizer:
        enable: true

        frame_accumulation:
            enable: false
            type: sliding_sum
            window_length_seconds: 1.0

        frame_binarization:
            enable: true
            type: global_threshold
            threshold: 0.5

        event_activity_processing:
            enable: true
            type: median_filtering
            window_length_seconds: 0.54

        event_post_processing:
            enable: true
            minimum_event_length_seconds: 0.1
            minimum_event_gap_second: 0.1

+--------------------------------+--------------------+----------------------------------------------------------------------+
| Field name                     | Value type         | Description                                                          |
+================================+====================+======================================================================+
| enable                         | bool               | Section selector                                                     |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **frame_accumulation**, Defining frame probability accumulation.                                                           |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable frame probability accumulation.                               |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Operator type used to accumulate.                                    |
|                                | {sliding_sum |     |                                                                      |
|                                | sliding_mean |     |                                                                      |
|                                | sliding_median }   |                                                                      |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| window_length_seconds          | float              | Window length in seconds for sliding accumulation.                   |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **frame_binarization**, Defining frame probability binarization.                                                           |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable frame probability binarization.                               |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Type of binarization:                                                |
|                                | {frame_max |       |                                                                      |
|                                | global_threshold } | - ``frame_max``, each frame is treated individually, max of each     |
|                                |                    |   frame is set to one, all others to zero.                           |
|                                |                    | - ``global_threshold``, global threshold, all values over the        |
|                                |                    |   threshold are set to one.                                          |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| threshold                      | float              | Threshold value. Set to null if not used.                            |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **event_activity_processing**, Event activity processing per frame.                                                        |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable activity processing.                                          |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| type                           | string             | Type of decision:                                                    |
|                                | {median_filtering} |                                                                      |
|                                |                    | - ``median_filtering``, median filtering of decision inside window.  |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| window_length_seconds          | float              | Length of sliding window in seconds for activity processing.         |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **event_post_processing**, Event post processing per event.                                                                |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable event processing.                                             |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| minimum_event_length_seconds   | float              | Minimum allowed event length. Shorter events will be removed.        |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| minimum_event_gap_second       | float              | Minimum allowed gap between events. Smaller gaps between events      |
|                                |                    | will cause events to be merged together.                             |
+--------------------------------+--------------------+----------------------------------------------------------------------+


Evaluator
^^^^^^^^^

This section defines the evaluation stage of the system.

Example section:

.. code-block:: yaml

    evaluator:
        enable: true
        show_details: false

        saving:
            enable: true
            filename: eval_[{parameter_hash}].yaml

+--------------------------------+--------------------+----------------------------------------------------------------------+
| Field name                     | Value type         | Description                                                          |
+================================+====================+======================================================================+
| enable                         | bool               | Section selector                                                     |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| show_details                   | bool               | Show more detailed metrics (class-wise, scene-wise, event-wise)      |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **saving**, Saving evaluation results                                                                                      |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| enable                         | bool               | Enable result saving into yaml-file.                                 |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| filename                       | string             | Filename for the evalution results. Following magic fields can be    |
|                                |                    | used:                                                                |
|                                |                    |                                                                      |
|                                |                    | - ``{parameter_hash}``                                               |
|                                |                    | - ``{parameter_set}``                                                |
|                                |                    | - ``{dataset_name}``                                                 |
|                                |                    |                                                                      |
+--------------------------------+--------------------+----------------------------------------------------------------------+


Logging
^^^^^^^

This section defines the system logging.

Example section:

.. code-block:: yaml

    logging:
        enable: true
        colored: true

        parameters:
            version: 1
            disable_existing_loggers: false
            formatters:
                simple:
                    format: "[%(levelname)-8s] %(message)s"
                normal:
                    format: "%(asctime)s\t[%(name)-20s]\t[%(levelname)-8s]\t%(message)s"
                extended:
                    format: "[%(asctime)s] [%(name)s]\t [%(levelname)-8s]\t %(message)s \t(%(filename)s:%(lineno)s)"
            handlers:
                console:
                    class: logging.StreamHandler
                    level: DEBUG
                    formatter: simple
                    stream: ext://sys.stdout

                info_file_handler:
                    class: logging.handlers.RotatingFileHandler
                    level: INFO
                    formatter: normal
                    filename: task1.info.log
                    maxBytes: 10485760
                    backupCount: 20
                    encoding: utf8

                debug_file_handler:
                    class: logging.handlers.RotatingFileHandler
                    level: DEBUG
                    formatter: normal
                    filename: task1.debug.log
                    maxBytes: 10485760
                    backupCount: 20
                    encoding: utf8

                error_file_handler:
                    class: logging.handlers.RotatingFileHandler
                    level: ERROR
                    formatter: extended
                    filename: task1.errors.log
                    maxBytes: 10485760
                    backupCount: 20
                    encoding: utf8

            loggers:
                my_module:
                    level: ERROR
                    handlers: [console]
                    propagate: no

            root:
                level: INFO
                handlers: [console, error_file_handler, info_file_handler, debug_file_handler]

+--------------------------------+--------------------+----------------------------------------------------------------------+
| Field name                     | Value type         | Description                                                          |
+================================+====================+======================================================================+
| enable                         | bool               | Enable logging                                                       |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| colored                        | bool               | Enable colored logging when printing it on console.                  |
+--------------------------------+--------------------+----------------------------------------------------------------------+
| **parameters**, Logging parameters ``logging.config.dictConfig(parameters)``, see                                          |
| `documentation <https://docs.python.org/2/library/logging.config.html#dictionary-schema-details>`_.                        |
+--------------------------------+--------------------+----------------------------------------------------------------------+

