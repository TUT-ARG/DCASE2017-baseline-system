.. _framework:
.. figure:: _images/dcase_framework.png

Introduction
============

The baseline system is built on top of the DCASE Framework, a collection of utility classes for DCASE related research.
The framework provides tools for system parameter handling, feature extraction, data storage, basic model learning.
In addition to the utility classes, the framework provides application classes specialized for sound classification
and sound event detection. Application classes handle the full pipeline of the common steps: feature extraction,
feature normalization, acoustic model training, system testing and system evaluation. These application classes
can be extended to accommodate different research problems.

Main idea of the framework is to ease rapid experimenting in tasks related to DCASE. Most of the system design can be
done through parameter files, and new features can be introduced to the framework through class inheritance. This
allows users to keep their research code outside dcase_framework code base.

Main design principles:

- Concentrate on system blocks needed for basic approaches, and introduce more advanced blocks in moderate pace.
- Use object-oriented design and design class APIs to allow class extension through class inheritance.
- Prefer readable code over high optimized code, avoid one-liners and hackish code.
- Wrap data into container classes when possible.
- Use parametrization extensively.

The framework is actively developed, and the class and function API might change during the development.
Once the framework has reached reasonable level of stability, it will be released as a standalone python module to ease
usage outside DCASE challenge.


Training process
^^^^^^^^^^^^^^^^

.. figure:: _images/dcase_framework_training.svg
    :width: 100%

Testing process
^^^^^^^^^^^^^^^

.. figure:: _images/dcase_framework_testing.svg
    :width: 100%