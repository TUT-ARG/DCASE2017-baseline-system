.. _applications:
.. |task1| image:: _images/task1_icon.png
.. |task2| image:: _images/task2_icon.png
.. |task3| image:: _images/task3_icon.png
.. |task4| image:: _images/task4_icon.png

Applications
============

.. _task1:

|task1| Acoustic scene classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The goal of acoustic scene classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded — for example "park", "home", "office".

.. figure:: _images/task1_overview.png
    :width: 70%
    :align: center

    System overview for acoustic scene classification application.

`More information on DCASE2017 Task 1 page. <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-acoustic-scene-classification>`_

Results
*******

**TUT Acoustic Scenes 2017, Development**

*Average accuracy of file-wise classification.*

+------------------------+------------+--------+--------+--------+--------+
|                        | Overall    | Folds                             |
+------------------------+------------+--------+--------+--------+--------+
| System                 | Accuracy   | 1      | 2      | 3      | 4      |
+========================+============+========+========+========+========+
| MLP based system,      | 74.8 %     | 75.2%  | 75.3 % | 77.3 % | 71.3 % |
| **DCASE2017 baseline** |            |        |        |        |        |
+------------------------+------------+--------+--------+--------+--------+
| GMM based system       | 74.1 %     | 74.0 % | 76.0 % | 73.1 % | 73.2 % |
+------------------------+------------+--------+--------+--------+--------+

Scene class-wise results

+------------------------+------------+------------+
|                        | System                  |
+------------------------+------------+------------+
| Scene class            | MLP        | GMM        |
+========================+============+============+
| beach                  | 75.3       | 75.0       |
+------------------------+------------+------------+
| bus                    | 71.8       | 84.3       |
+------------------------+------------+------------+
| cafe/restaurant        | 57.7       | 81.7       |
+------------------------+------------+------------+
| car                    | 97.1       | 91.0       |
+------------------------+------------+------------+
| city center            | 90.7       | 91.0       |
+------------------------+------------+------------+
| forest path            | 79.5       | 73.4       |
+------------------------+------------+------------+
| grocery store          | 58.7       | 67.9       |
+------------------------+------------+------------+
| home                   | 68.6       | 71.4       |
+------------------------+------------+------------+
| library                | 57.1       | 63.5       |
+------------------------+------------+------------+
| metro station          | 91.7       | 81.4       |
+------------------------+------------+------------+
| office                 | 99.7       | 97.1       |
+------------------------+------------+------------+
| park                   | 70.2       | 39.1       |
+------------------------+------------+------------+
| residential area       | 64.1       | 74.7       |
+------------------------+------------+------------+
| train                  | 58.0       | 41.0       |
+------------------------+------------+------------+
| tram                   | 81.7       | 79.2       |
+------------------------+------------+------------+
| **Overall**            | 74.8       | 74.1       |
+------------------------+------------+------------+

To reproduce the results run::

    make -C docker/ task1

See more about :ref:`reproducibility <reproducibility>`.

*Results calculated with Python 2.7.13, Keras 2.0.2, and Theano 0.9.0*


.. _task2:

|task2| Detection of rare sound events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task focuses on detection of rare sound events in artificially created mixtures. The goal is to output for each test file the information on whether the target sound event has been detected, including the textual label, onset and offset of the detected sound event.

.. figure:: _images/task2_overview.png
    :width: 70%
    :align: center

    System overview for detection of rare sound events application.

`More information on DCASE2017 Task 2. <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-rare-sound-event-detection>`_

Results
*******

**TUT Rare Sound Events 2017, Development**

*Event-based metric*

+------------------------+------------+---------+
|                        | Event-based metrics  |
+------------------------+------------+---------+
| System                 | ER         | F-score |
+========================+============+=========+
| MLP based system,      | 0.53       | 72.7 %  |
| **DCASE2017 baseline** |            |         |
+------------------------+------------+---------+
| GMM based system       | 0.55       | 72.5 %  |
+------------------------+------------+---------+

Event class-wise results

+------------------------+--------+----------+--------+----------+
|                        | System                                |
+------------------------+--------+----------+--------+----------+
|                        | MLP               | GMM               |
+------------------------+--------+----------+--------+----------+
| Event class            | ER     | F-score  | ER     | F-score  |
+========================+========+==========+========+==========+
| babycry                | 0.67   | 72.0     | 0.77   | 67.6     |
+------------------------+--------+----------+--------+----------+
| glassbreak             | 0.22   | 88.5     | 0.35   | 82.8     |
+------------------------+--------+----------+--------+----------+
| gunshot                | 0.69   | 57.4     | 0.54   | 67.2     |
+------------------------+--------+----------+--------+----------+
| **Overall**            | 0.53   | 72.7     | 0.55   | 72.5     |
+------------------------+--------+----------+--------+----------+

To reproduce these results run::

    make -C docker/ task2

See more about :ref:`reproducibility <reproducibility>`.

*Results calculated with Python 2.7.13, Keras 2.0.2, and Theano 0.9.0*

More details on the metrics calculation can be found in:

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, "*Metrics for polyphonic sound event detection*", Applied Sciences, 6(6):162, 2016 [`HTML <http://www.mdpi.com/2076-3417/6/6/162>`_][`PDF <http://www.mdpi.com/2076-3417/6/6/162/pdf>`_]

.. _task3:

|task3| Sound event detection in real life audio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task evaluates performance of the sound event detection systems in multisource conditions similar to our everyday life, where the sound sources are rarely heard in isolation. In this task, there is no control over the number of overlapping sound events at each time, not in the training nor in the testing audio data.

.. figure:: _images/task3_overview.png
    :width: 70%
    :align: center

    System overview for sound event detection in real life audio application.

`More information on DCASE2017 Task 3. <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-sound-event-detection-in-real-life-audio>`_

Results
*******

**TUT Sound Events 2017, Development**

*Segment-based metric*

+------------------------+------------+----------+
|                        | Segment-based metrics |
+------------------------+------------+----------+
| System                 | ER         | F-score  |
+========================+============+==========+
| MLP based system,      | 0.69       | 56.7 %   |
| **DCASE2017 baseline** |            |          |
+------------------------+------------+----------+
| GMM based system       | 0.71       | 52.1 %   |
+------------------------+------------+----------+

Event class-wise metrics

+------------------------+--------+----------+--------+----------+
|                        | System                                |
+------------------------+--------+----------+--------+----------+
|                        | MLP               | GMM               |
+------------------------+--------+----------+--------+----------+
| Event class            | ER     | F-score  | ER     | F-score  |
+========================+========+==========+========+==========+
| brakes squeaking       | 0.98   | 4.1      | 1.06   | 13.6     |
+------------------------+--------+----------+--------+----------+
| car                    | 0.57   | 74.1     | 0.60   | 66.4     |
+------------------------+--------+----------+--------+----------+
| children               | 1.35   | 0.0      | 1.54   | 0.0      |
+------------------------+--------+----------+--------+----------+
| large vehicle          | 0.90   | 50.8     | 0.98   | 38.0     |
+------------------------+--------+----------+--------+----------+
| people speaking        | 1.25   | 18.5     | 1.23   | 28.5     |
+------------------------+--------+----------+--------+----------+
| people walking         | 0.84   | 55.6     | 0.61   | 65.6     |
+------------------------+--------+----------+--------+----------+


To reproduce these results run::

    make -C docker/ task3

See more about :ref:`reproducibility <reproducibility>`.

*Results calculated with Python 2.7.13, Keras 2.0.2, and Theano 0.9.0*

More details on the metrics calculation can be found in:

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, "*Metrics for polyphonic sound event detection*", Applied Sciences, 6(6):162, 2016 [`HTML <http://www.mdpi.com/2076-3417/6/6/162>`_][`PDF <http://www.mdpi.com/2076-3417/6/6/162/pdf>`_]