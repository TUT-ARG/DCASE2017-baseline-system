.. _framework:
.. figure:: _images/dcase_framework.png

Introduction
============

The baseline system is built on top of the DCASE Framework, a collection of utility classes for DCASE related research. The framework provides tools for system parameter handling, feature extraction, data storage, basic model learning. In addition to the utility classes, the framework provides application classes specialized for sound classification and sound event detection. Application classes handle the full pipeline of the common steps: feature extraction, feature normalization, acoustic model training, system testing and system evaluation. These application classes can be extended easily to accommodate different research problems.


Training process
^^^^^^^^^^^^^^^^

.. figure:: _images/dcase_framework_training.svg
    :width: 100%

Testing process
^^^^^^^^^^^^^^^

.. figure:: _images/dcase_framework_testing.svg
    :width: 100%