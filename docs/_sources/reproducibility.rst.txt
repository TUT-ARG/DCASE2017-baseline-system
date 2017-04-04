.. _reproducibility:

Reproducibility
===============

The baseline system is based on `Keras <https://keras.io/>`_ while using `Theano <http://deeplearning.net/software/theano/>`_ as backend.
Theano can be used seamlessly either on CPU or on GPU, however, the learned neural networks from these two modes will not be exactly the same.
Theano can produce consistent results on **same computer setup** when used in CPU mode, if numpy random seed is fixed prior to importing Keras. In GPU mode, consistent randomization values are generally hard to achieve due to the heavy usage of parallel computing. Fortunately, results are usually very close each other and within acceptable limits to enable informed system development. However, the role of a baseline system is to have results which are reproducible on most of the normal computer setups.

In order to get consistent results across **different computer setups** when using Keras and Theano, the following aspects should be taken into account:

- Use exactly the same Python version, and Python library versions. Use pip freeze command and ``requirements.txt`` to document the libraries.
- Do not rely on dictionary keys to be in fixed order. Sort keys before usage to ensure consistent data processing order, or use `OrderedDict <https://docs.python.org/2/library/collections.html#collections.OrderedDict>`_ data type.
- Set ``random.seed`` at the beginning of the application.
- Set ``numpy.random.seed`` before Keras is imported.
- Use Theano in CPU mode.
- Use same BLAS library for the CPU computation, use either ATLAS or Inter Math Kernel (in compatibile mode) library to get same results regardless of underlying CPU architecture.
- Run the system in single thread mode to avoid inconsistencies caused by threading.


.. _blas:

BLAS libraries
^^^^^^^^^^^^^^

When running Theano in CPU mode, the computation relies on system level `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ library (this is not a Python library).
BLAS libraries usually have implementations for mathematical operations which are optimized for speed, not for accuracy, and the particular implementation is selected based on the CPU architecture the library is run. This approach can produce slightly deviating results from CPU to CPU.

Theano supports three common BLAS libraries: `ATLAS <http://math-atlas.sourceforge.net/>`_, `OpenBLAS <http://www.openblas.net/>`_, and `Intel Math Kernel Library (MKL) <https://software.intel.com/en-us/intel-mkl>`_. In our tests, we found that these libraries can produce deviating results when run, for example, on Intel processors or on AMD processors. The ATLAS and MKL (with correct flags) libraries were found to give consistent results across different computers supporting SSE2 processor instruction set, whereas OpenBLAS failed to produce consistent results.

Intel Math Kernel
-----------------

Intel Math Kernel (MKL) is usually faster than ATLAS, and therefore it was selected to be used for the baseline system. MKL is a commercial library which can be downloaded for free with `Named-User Licence from Intel <https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2>`_. Easiest way to get access to MKL for Python development is to use `Anaconda Python distribution <https://www.continuum.io/why-anaconda>`_ which comes readily with MKL `installation <https://www.continuum.io/blog/developer-blog/anaconda-25-release-now-mkl-optimizations>`_.

To get MKL to behave consistently across computer setups, conditional numerical reproducibility should be set into compatible mode with ``MKL_CBWR=COMPATIBLE`` environmental variable (see more `here <https://software.intel.com/en-us/node/528408>`_). Furthermore, threading should be disabled by using ``MKL_NUM_THREADS=1`` and ``MKL_DYNAMIC=FALSE`` environmental variables.

Running the baseline system
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The baseline system is delivered with `Docker <https://www.docker.com/>`_ software container which will allow the system to run inside a virtual operating system environment to produce consistent system output. Docker provides operating system level virtualization on Linux and Windows platforms, and the provided Docker container contains a minimal linux OS, additional system libraries, and minimal Anaconda Python installation with required Python libraries.

To install Docker Community Edition (Docker CE), follow the instructions from `Docker documentation <https://docs.docker.com/engine/installation/>`_.

There is a ``Makefile`` in ``docker`` directory to help usage of the Docker container.
When container is launched first time, Docker will create the container image by downloading and installing the needed libraries (defined in ``Dockerfile``).

To generate the baseline system results shown in :ref:`applications page <applications>` run (example given for Task 1)::

    make -C docker/ task1

or::

    cd docker
    make task1

To open bash shell inside the container environment::

    make -C docker/ bash

or::

    cd docker
    make bash


