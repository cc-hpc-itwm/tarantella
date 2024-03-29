.. _installation-label:

Installation
============

Tarantella needs to be built `from source <https://github.com/cc-hpc-itwm/tarantella>`_.
Since Tarantella is built on top of `TensorFlow <https://www.tensorflow.org/>`_,
you will require a recent version of it. Additionally, you will need an installation of
the open-source communication libraries `GaspiCxx <https://github.com/cc-hpc-itwm/GaspiCxx>`_
and `GPI-2 <http://www.gpi-site.com/>`_,
which Tarantella uses to implement distributed training.

Lastly, you will need `pybind11 <https://github.com/pybind/pybind11>`_, which is required
for Python and C++ inter-communication.

In the following we will look at the required steps in detail.

Installing dependencies
-----------------------

Compiler and build system
^^^^^^^^^^^^^^^^^^^^^^^^^

Tarantella can be built using a recent `gcc <https://gcc.gnu.org/>`_
compiler with support for C++17 (starting with ``gcc 7.4.0``).
You will also need the build tool `CMake <https://cmake.org/>`_ (from version ``3.12``).

Installing TensorFlow
^^^^^^^^^^^^^^^^^^^^^

First you will need to install TensorFlow.
Supported versions start at ``Tensorflow 2.4``, and they can be installed in a conda
environment using pip, as recommended on the
`TensorFlow website <https://www.tensorflow.org/install>`_.

In order to do that, first install `conda <https://docs.conda.io/en/latest/>`_ on your system.
Then, create and activate an environment for Tarantella:

.. code-block:: bash

  conda create -n tarantella
  conda activate tarantella

Now, you can install the latest supported TensorFlow version with:

.. code-block:: bash

  conda install python=3.9
  pip install --upgrade tensorflow==2.9.*

Tarantella requires at least Python ``3.7``. Make sure the selected version also matches
the `TensorFlow requirements <https://www.tensorflow.org/install>`_.

.. _installation-pybind11-label:

Installing pybind11
^^^^^^^^^^^^^^^^^^^

The next dependency you will need to install is
`pybind11 <https://pybind11.readthedocs.io/en/stable/index.html>`__,
which is available through pip and conda.
We recommend installing ``pybind11`` via conda:

.. code-block:: bash

  conda install pybind11 -c conda-forge


.. _gpi2-install-label:

Installing GPI-2
^^^^^^^^^^^^^^^^

Next, you will need to download, compile and install the GPI-2 library.
GPI-2 is an API for high-performance, asynchronous communication for large scale
applications, based on the
`GASPI (Global Address Space Programming Interface) standard <http://www.gaspi.de>`_.

The currently supported versions start with ``1.5``, and they need to be built with
position independent flags (``-fPIC``).
To download the required version, clone the
`GPI-2 git repository <https://github.com/cc-hpc-itwm/GPI-2.git>`_
and checkout the latest ``tag``:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/GPI-2.git
  cd GPI-2
  git fetch --tags
  git checkout -b v1.5.1 v1.5.1

Now, use `autotools <https://www.gnu.org/software/automake/>`_ to configure and compile the code:

.. code-block:: bash

  ./autogen.sh 
  export GPI2_INSTALLATION_PATH=/your/gpi2/installation/path
  CFLAGS="-fPIC" CPPFLAGS="-fPIC" ./configure --with-ethernet --prefix=${GPI2_INSTALLATION_PATH}
  make -j$(nproc)

where ``${GPI2_INSTALLATION_PATH}`` needs to be replaced with the path where you want to install
GPI-2. Note the ``--with-ethernet`` option, which will use standard TCP sockets for communication.
This is the correct option for laptops and workstations.

.. _gpi-build-infiniband-label:

In case you want to use Infiniband, replace the above option with ``--with-infiniband``.
Now you are ready to install GPI-2 with:

.. code-block:: bash

  make install
  export PATH=${GPI2_INSTALLATION_PATH}/bin:$PATH

where the last two commands make the library visible to your system.
If required, GPI-2 can be removed from the target directory by using ``make uninstall``.

.. _gaspicxx-install-label:

Installing GaspiCxx
^^^^^^^^^^^^^^^^^^^

`GaspiCxx <https://github.com/cc-hpc-itwm/GaspiCxx>`_ is a C++ abstraction layer built
on top of the GPI-2 library, designed to provide easy-to-use point-to-point and collective
communication primitives.
Tarantella's communication layer is based on GaspiCxx and its
`PyGPI <https://github.com/cc-hpc-itwm/GaspiCxx/blob/v1.2.0/src/python/README.md>`_ API for Python.
Currently we support GaspiCxx version v1.2.0.

To install GaspiCxx and PyGPI, first download the latest release from the
`git repository <https://github.com/cc-hpc-itwm/GaspiCxx>`_:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/GaspiCxx.git
  cd GaspiCxx
  git fetch --tags
  git checkout -b v1.2.0 v1.2.0

GaspiCxx requires an already installed version of GPI-2, which should be detected at
configuration time (as long as ``${GPI2_INSTALLATION_PATH}/bin`` is added to the current
``${PATH}`` as shown :ref:`above <gpi2-install-label>`).

Compile and install the library as follows, making sure the previously created conda
environment is activated:

.. code-block:: bash

  conda activate tarantella

  mkdir build && cd build
  export GASPICXX_INSTALLATION_PATH=/your/gaspicxx/installation/path
  cmake -DBUILD_PYTHON_BINDINGS=ON    \
        -DBUILD_SHARED_LIBS=ON        \
        -DCMAKE_INSTALL_PREFIX=${GASPICXX_INSTALLATION_PATH} ../
  make -j$(nproc) install

where ``${GASPICXX_INSTALLATION_PATH}`` needs to be set to the path where you want to install
the library.

SSH key-based authentication
----------------------------

In order to use Tarantella on a cluster, make sure you can ssh between nodes
without password. For details, refer to the :ref:`FAQ section <faq-label>`.
In particular, to test Tarantella on your local machine, make sure
you can ssh to ``localhost`` without password.

Building Tarantella from source
-------------------------------

With all dependencies installed, we can now download, configure and compile Tarantella.
To download the source code, simply clone the
`GitHub repository <https://github.com/cc-hpc-itwm/tarantella.git>`__:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/tarantella.git
  cd tarantella
  git checkout tags/v0.9.0 -b v0.9.0

Next, we need to configure the build system using CMake.
For a standard out-of-source build, we create a separate ``build`` folder and run ``cmake``
in it:

.. code-block:: bash

  conda activate tarantella

  cd tarantella
  mkdir build && cd build
  export TARANTELLA_INSTALLATION_PATH=/your/installation/path
  cmake -DCMAKE_INSTALL_PREFIX=${TARANTELLA_INSTALLATION_PATH} \
        -DCMAKE_PREFIX_PATH=${GASPICXX_INSTALLATION_PATH} ../

This will configure your installation to use the previously installed GPI-2 and GaspiCxx
libraries. To install Tarantella on a cluster equipped with Infiniband capabilities,
make sure that GPI-2 is installed with Infiniband support as shown
:ref:`here <gpi-build-infiniband-label>`.

Now, we can compile and install Tarantella to ``TARANTELLA_INSTALLATION_PATH``:

.. code-block:: bash

  make -j$(nproc) install
  export PATH=${TARANTELLA_INSTALLATION_PATH}/bin:${PATH}


[Optional] Building and running tests
-------------------------------------

In order to build Tarantella with tests, you will also need to install
`Boost <https://www.boost.org/>`_
(for C++ tests), and `pytest <https://www.pytest.org/>`_ (for Python tests).
Additionally, the `PyYAML <https://pypi.org/project/PyYAML/>`_ and
`NetworkX <https://networkx.org/>`_ libraries are required by some tests.

To install boost with the required `devel`-packages, under Ubuntu you can use

.. code-block:: bash

  sudo apt install libboost-all-dev

while in Fedora you can use

.. code-block:: bash

  sudo dnf install boost boost-devel

The other dependencies can be installed in the existing conda environment:

.. code-block:: bash

  pip install -U pytest
  pip install PyYAML==3.13
  conda install networkx


After having installed these libraries, make sure to configure Tarantella with testing switched on:

.. code-block:: bash

  cd tarantella
  mkdir build && cd build
  export LD_LIBRARY_PATH=`pwd`:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${GPI2_INSTALLATION_PATH}/lib64:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${GASPICXX_INSTALLATION_PATH}/lib:${LD_LIBRARY_PATH}

  export PYTHONPATH=`pwd`:${PYTHONPATH}
  export PYTHONPATH=${GASPICXX_INSTALLATION_PATH}/lib:${PYTHONPATH}

  cmake -DENABLE_TESTING=ON ../

Now you can compile Tarantella and run its tests in the ``build`` directory:

.. code-block:: bash

  make -j$(nproc)
  ctest

[Optional] Building documentation
---------------------------------

If you would like to build `the documentation <https://tarantella.readthedocs.io/en/latest/>`_
locally, run the following ``cmake`` command

.. code-block:: bash

  cmake -DCMAKE_INSTALL_PREFIX=${TARANTELLA_INSTALLATION_PATH} -DBUILD_DOCS=ON ..

before compiling.
This requires you to have `Sphinx <https://www.sphinx-doc.org/en/master/>`_ installed:

.. code-block:: bash

  pip install -U sphinx
