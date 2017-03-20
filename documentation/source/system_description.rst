.. _system_description:

System description
==================

.. figure:: _images/system_approach.svg
    :width: 100%

    System block diagram.

MLP based system, DCASE2017 baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A multilayer perceptron based system is selected as baseline system for DCASE2017. The main structure of the system is
close to the current state-of-art systems which are based on recurrent neural networks (RNN) and
convolutional neural networks (CNN), and therefore it provides a good starting point for further development. The system is
implemented around `Keras <https://keras.io/>`_, a high-level neural networks API written in Python. Keras works on top
of multiple computation backends, of which `Theano <http://deeplearning.net/software/theano/>`_ was chosen for this system.

System details:

- *Acoustic features*: Log mel-band energies extracted in 40ms windows with 20ms hop size.
- *Machine learning*: neural network approach using multilayer perceptron (MLP) type of network (2 layers with 50 neurons each, and 20% dropout between layers).

**System hyperparameters**

+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Application          | Scene           | Binary          | Multiclass      |                                        |
|                      | Classification  | SED             | SED             |                                        |
|                      | (Task 1)        | (Task 2)        | (Task 3)        |                                        |
+======================+=================+=================+=================+========================================+
| **Acoustic features**  (`Librosa 0.5 <https://librosa.github.io/librosa/>`_)                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Type                 | Log mel energies                                                                             |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Window length        | 40 ms                                                                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Hop length           | 20 ms                                                                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Mel bands            | 40                                                                                           |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| **Feature vector**                                                                                                  |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Aggregation          | 5 frame context                                                                              |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Length               | 120                                                                                          |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| **Neural Network** (`Keras 2.0 <https://keras.io>`_ +                                                               |
| `Theano <http://deeplearning.net/software/theano/>`_ with CPU device)                                               |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Layers               | 2 `Dense layers <https://keras.io/layers/core/>`_                                            |
|                      | with 20% `Dropout layers <https://keras.io/layers/core/>`_ between.                          |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Hidden units per     | 50                                                                                           |
| layer                |                                                                                              |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Initialization       | `Uniform <https://keras.io/initializers/>`_                                                  |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Activation           | `ReLU <https://keras.io/activations/>`_                                                      |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Output layer type    | Softmax         | Sigmoid         | Sigmoid         |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Optimizer            | `Adam <https://keras.io/optimizers/>`_                                                       |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Learning rate        | 0.001                                               |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Epochs               | 200                                                                                          |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Batch size           | 256                                                                                          |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Number of parameters | 12906                                                                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Decision             | Binarization per| Binarization per frame +          |                                        |
|                      | frame + majority| sliding median filtering          |                                        |
|                      | vote            |                                   |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Median filter window |                 | 0.54 sec        |  0.54 sec       |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Binarization         | 0.5                                                 |                                        |
| threshold            |                                                     |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+

GMM based approach
^^^^^^^^^^^^^^^^^^

A secondary system based on Gaussian mixture models is also included in the baseline system
in order to enable comparison to the traditional systems presented in the literature.
The implementation of the GMM-based system is very similar to the baseline system used in
DCASE2016 Challenge for Task 1 and Task 3. See more details about the system used for DCASE2016:

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, "*TUT database for acoustic scene classification and sound event detection*". In 24th European Signal Processing Conference 2016 (EUSIPCO 2016). Budapest, Hungary, 2016. [`PDF <http://www.cs.tut.fi/~mesaros/pubs/mesaros_eusipco2016-dcase.pdf>`_]

System details:

- *Acoustic features*:  20 MFCC static coefficients (including 0th) + 20 delta MFCC coefficients + 20 acceleration MFCC coefficients = 60 values, calculated in 40 ms analysis window with 50% hop size
- *Machine learning*: Gaussian mixture models, 16 Gaussians per class model

**System hyperparameters**

+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Application          | Scene           | Binary          | Multiclass      |                                        |
|                      | Classification  | SED             | SED             |                                        |
|                      | (Task 1)        | (Task 2)        | (Task 3)        |                                        |
+======================+=================+=================+=================+========================================+
| **Acoustic features**  (`Librosa 0.5 <https://librosa.github.io/librosa/>`_)                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Type                 | MFCC (static, delta, acceleration)                  |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Window length        | 40 ms                                               |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Hop length           | 20 ms                                               |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Mel bands            | 40                                                                                           |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Number of            | 20                                                  |                                        |
| coefficients         |                                                     |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Delta window         | 9                                                   |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| **Feature vector**                                                         |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Aggregation          | mfcc+delta+acc  | mfcc (0th omitted) + delta+acc    |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Length               | 60              | 59                                |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| **Gaussian Mixtures** (`Sklearn GaussianMixture                                                                     |
| <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html>`_)                          |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Number of Gaussians  | 16                                                  |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Covariance           | diagonal                                            |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Number of parameters | 1936            | 1904                              |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Modelling            | One model       | Model pair per event class,       |                                        |
|                      | per scene       | negative model and positive model.|                                        |
|                      | class           |                                   |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Decision             | Likelihood      | Sliding likelihood accumulation + |                                        |
|                      | accumulation +  | likelihood ratio + thresholding   |                                        |
|                      | maximum         |                                   |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Accumulation window  | Signal length   | 0.5 sec         |  1.0 sec        |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+
| Decision threshold   |                 | 200             | 100             |                                        |
+----------------------+-----------------+-----------------+-----------------+----------------------------------------+

Processing blocks
^^^^^^^^^^^^^^^^^

.. figure:: _images/system_processing_blocks.svg
    :width: 90%
    :align: left

    Processing blocks of the system.

The system implements the following basic processing blocks:

1. **Initialization**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.initialize`

  - Prepares the dataset:

    - Downloads the dataset from the Internet if needed
    - Extracts the dataset package if needed
    - Makes sure that the meta files are appropriately formatted


2. **Feature extraction**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.feature_extraction`

  - Goes through all the training material and extracts the acoustic features
  - Features are stored file-by-file on the local disk (pickle files)

3. **Feature normalization**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.feature_normalization`

  - Goes through the training material in evaluation folds, and calculates global mean and std of the data per fold.
  - Stores the normalization factors per fold(pickle files)

4. **System training**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.system_training`

  - Loads normalizers
  - Loads training material file-by-file and forms the feature matrix, normalizes the matrix and optionally aggregates features
  - Trains the system with the features and metadata
  - Stores the trained acoustic models on the local disk (pickle files)

5. **System testing**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.system_testing`

  - Goes through the testing material and does the classification / detection
  - Stores the results (text files)

6. **System evaluation**, see :py:meth:`~dcase_framework.application_core.AcousticSceneClassificationAppCore.system_evaluation`

  - Reads the ground truth and the output of the system and calculates evaluation metrics

