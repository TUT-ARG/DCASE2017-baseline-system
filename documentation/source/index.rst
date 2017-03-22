.. figure:: _images/dcase2017_baseline.png
.. |task1| image:: _images/task1_icon.png
.. |task2| image:: _images/task2_icon.png
.. |task3| image:: _images/task3_icon.png
.. |task4| image:: _images/task4_icon.png

.. |email_toni| image:: _images/email_icon.png
    :align: middle
    :width: 24
    :target: mailto:toni.heittola@tut.fi
.. |email_aleksandr| image:: _images/email_icon.png
    :align: middle
    :width: 24
    :target: mailto:aleksandr.diment@tut.fi
.. |email_annamaria| image:: _images/email_icon.png
    :align: middle
    :width: 24
    :target: mailto:annamaria.mesaros@tut.fi

.. |home_toni| image:: _images/home_icon.png
    :align: middle
    :width: 24
    :target: http://www.cs.tut.fi/~heittolt/
.. |home_aleksandr| image:: _images/home_icon.png
    :align: middle
    :width: 24
    :target: http://www.cs.tut.fi/~diment/
.. |home_annamaria| image:: _images/home_icon.png
    :align: middle
    :width: 24
    :target: http://www.cs.tut.fi/~mesaros/
.. |git_toni| image:: _images/git_icon.png
    :align: middle
    :width: 24
    :target: https://github.com/toni-heittola

`Audio Research Group / Tampere University of Technology <http://arg.cs.tut.fi/>`_

**Authors**

+-----------------------+------------------------------------------------------------+-------------------+
| **Toni Heittola**     | Baseline system, DCASE Framework, Documentation            | |email_toni|      |
|                       |                                                            | |home_toni|       |
|                       |                                                            | |git_toni|        |
+-----------------------+------------------------------------------------------------+-------------------+
| **Aleksandr Diment**  | Dataset synthesis (Task 2)                                 | |email_aleksandr| |
|                       |                                                            | |home_aleksandr|  |
+-----------------------+------------------------------------------------------------+-------------------+
| **Annamaria Mesaros** | Documentation                                              | |email_annamaria| |
|                       |                                                            | |home_annamaria|  |
+-----------------------+------------------------------------------------------------+-------------------+

Introduction
============

This document describes the baseline system for the `Detection and Classification of Acoustic Scenes and Events 2017 (DCASE2017) challenge <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/>`_ tasks.

Baseline system
---------------

The baseline system is intended to lower the hurdle to participate to the DCASE challenges. It provides an entry-level
approach which is simple but relatively close to the state of the art systems to give reasonable performance for all
the tasks. High-end performance is left for the challenge participants to find.

In the baseline, one single low-level approach is shared across the tasks using application-specific extensions.
The main idea of this is to show the parallelism in the tasks settings, and how easily one can jump between
tasks during system development.

The main baseline system implements following approach:

- *Acoustic features*: Log mel-band energies extracted in 40ms windows with 20ms hop size.
- *Machine learning*: neural network approach using multilayer perceptron (MLP) type of network (2 layers with 50 neurons each, and 20% dropout between layers).

In addition to this, Gaussian mixture model based system is included for the comparison.

The system is developed for `Python 2.7 <https://www.python.org/>`_ and `Python 3.6 <https://www.python.org/>`_.
The system is tested to work with Linux and MacOS (10.12.) operating systems.

More about the :ref:`baseline system<system_description>`.


Applications
------------

Applications are specialized versions of the main system for specific tasks. The baseline system includes the following
applications tailored for `DCASE2017 challenge <http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/>`_:

|task1| Task 1, :ref:`Acoustic scene classification<task1>`

|task2| Task 2, :ref:`Detection of rare sound events<task2>`

|task3| Task 3, :ref:`Sound event detection in real life audio<task3>`

More about :ref:`applications<applications>`.

Getting started
---------------

1. Install requirements with command: ``pip install -r requirements.txt``
2. Run the application with default settings: ``python applications/task1.py``, ``python applications/task2.py``, and ``python applications/task3.py``

DCASE Framework
---------------

The baseline system is built on top of the DCASE Framework, a collection of utility classes designed to ease the DCASE related research process.
The framework provides tools for system parameter handling, acoustic feature extraction, data storage, acoustic model learning, and system evaluation.
In addition to the utility classes, the framework provides application classes to help build research code specialized for sound classification and sound event detection type of target applications.
Application classes handle the full development pipeline: feature extraction, feature normalization, model learning, model testing, and system evaluation.
These application classes can be extended easily to accommodate different research problems.

More about the :ref:`DCASE Framework<framework>`.


License
=======

The DCASE Framework and the baseline system is released **only for academic research** under `EULA <EULA.pdf>`_ from `Tampere University of Technology <http://www.tut.fi/en/home>`_.

See details from `EULA <EULA.pdf>`_.

Contents
========

.. toctree::
    :caption: Baseline system
    :name: baselinetoc
    :maxdepth: 1

    system_description
    applications

    install
    usage_tutorial
    parameterization

.. toctree::
    :caption: DCASE Framework
    :name: frameworktoc
    :maxdepth: 1

    framework

    extending_framework
    application_core
    parameters
    files
    datasets
    metadata
    features
    learners
    ui
    utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

