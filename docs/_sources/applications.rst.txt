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
| MLP based system,      | 73.3 %     | 74.1 % | 73.7 % | 72.6 % | 72.9 % |
| **DCASE2017 baseline** |            |        |        |        |        |
+------------------------+------------+--------+--------+--------+--------+
| GMM based system       | 73.5 %     | 74.6 % | 75.0 % | 73.3 % | 72.5 % |
+------------------------+------------+--------+--------+--------+--------+

To reproduce the results run::

    python task1.py -s dcase2017,gmm

*Results calculated with Python 2.7.13*


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
| MLP based system,      | 0.57       | 71.1 %  |
| **DCASE2017 baseline** |            |         |
+------------------------+------------+---------+
| GMM based system       | 0.61       | 70.6 %  |
+------------------------+------------+---------+

To reproduce these results run::

    python task2.py -s dcase2017,gmm

*Results calculated with Python 2.7.13*

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
| MLP based system,      | 0.76       | 51.8 %   |
| **DCASE2017 baseline** |            |          |
+------------------------+------------+----------+
| GMM based system       | 0.71       | 51.9 %   |
+------------------------+------------+----------+


To reproduce these results run::

    python task3.py -s dcase2017,gmm

*Results calculated with Python 2.7.13*

More details on the metrics calculation can be found in:

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, "*Metrics for polyphonic sound event detection*", Applied Sciences, 6(6):162, 2016 [`HTML <http://www.mdpi.com/2076-3417/6/6/162>`_][`PDF <http://www.mdpi.com/2076-3417/6/6/162/pdf>`_]