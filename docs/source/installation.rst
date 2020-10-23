Installation
============

Tarantella needs to be build `from source <https://github.com/cc-hpc-itwm/Tarantella>`_.
Since Tarantella is build on top of `TensorFlow 2 <https://www.tensorflow.org/>`_,
you will require a recent version of it. Additionally, you will need an installation of
the communication library `GPI-2 <http://www.gpi-site.com/>`_ which Tarantella uses
to communicated between processes.
Lastly you will need `pybind11 <https://github.com/pybind/pybind11>`_ which is required
for Python and C++ inter-communication.

In the following we will look at the required steps in detail.

Installing dependencies
-----------------------

Compiler and build system
^^^^^^^^^^^^^^^^^^^^^^^^^

Tarantella can be built using a recent `gcc <https://gcc.gnu.org/>`_
compiler (from version ``7.4.0``),
or a recent version of `clang <https://clang.llvm.org/>`_ (from version ``7.1.0``).
You will also need the build tool `CMake <https://cmake.org/>`_ (from version ``3.8``).

Installing GPI-2
^^^^^^^^^^^^^^^^

Next, you will need to download, compile and install the GPI-2 library.
The currently supported version is ``v1.4.0``, which needs to be build with
position independent flags (``-fPIC``).

To download the required version, clone the git repository and checkout the correct ``tag``:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/GPI-2.git
  git fetch --tags
  git checkout -b v1.4.0 v1.4.0

Now, use `autotools <https://www.gnu.org/software/automake/>`_ to configure and compile the code

.. code-block:: bash

  ./autogen.sh 
  CFLAGS="-fPIC" CPPFLAGS="-fPIC" ./configure --with-ethernet --prefix=${YOUR_INSTALLATION_PATH}
  make

where ``${YOUR_INSTALLATION_PATH}`` needs to be replaced by the path where you want to install
GPI-2. Note the ``--with-ethernet`` option, which will use standard TCP sockets for communication.
This is the correct option for laptops and workstations. In case you want to use Infiniband,
replace above option with ``--with-infiniband``.

Now you are ready to install GPI-2 with

.. code-block:: bash

  make install
  export PATH=${YOUR_INSTALLATION_PATH}/bin/:$PATH

where the last command makes the library visible to your ``PATH``.
If required, GPI-2 can be removed from the target directory by using ``make uninstall``.

Installing TensorFlow 2
^^^^^^^^^^^^^^^^^^^^^^^

Next you will need to install a version of TensorFlow 2.
Tarantella supports TensorFlow versions ``2.0`` to ``2.3``.
Either version can be installed in a conda environment using pip,
as recommended on the `TensorFlow website <https://www.tensorflow.org/install>`_.

In order to do that, first install `conda <https://docs.conda.io/en/latest/>`_ on your system.
Then, create and activate an environment for Tarantella:

.. code-block:: bash

  conda create tarantella
  conda activate tarantella

Now, you can install the latest supported TensorFlow version with

.. code-block:: bash

  conda install python=3.7
  pip install --upgrade tensorflow==2.3

Installing pybind11
^^^^^^^^^^^^^^^^^^^

The last dependency you will need to install is pybind11.
pybind11 is available through pip and conda. However, the pip-package does not seem
to include the CMake package, which is why we recommend installing pybind11 via conda:

.. code-block:: bash

  conda install pybind11 -c conda-forge

SSH to localhost
----------------

In order to test Tarantella on your local machine, make sure you can ssh to ``localhost``
without password. For details, we refer to the :ref:`FAQ section <faq-label>`.

Building Tarantella from source
-------------------------------

With all dependencies installed, we can now download, configure and compile Tarantella.
To download the source code, simply clone the GitHub repository:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/Tarantella.git

Next, we need to configure the build system using CMake.
For a standard out-of-source build, we create a separate ``build`` folder and run ``cmake``
in it:

.. code-block:: bash

  cd Tarantella
  mkdir build && cd build
  cmake ..

Now, we can compile and install Tarantella:

.. code-block:: bash

  make
  make install

.. todo::

  * add install directory above
  * what is a good default?

[Optional] Building and running tests
-------------------------------------

In order to build Tarantella with tests, you will also need to install
`Boost <https://www.boost.org/>`_
(for C++ tests), and `pytest <https://www.pytest.org/>`_ (for Python tests).

To install boost with the required `devel`-packages, under Ubuntu you can use

.. code-block:: bash

  sudo apt install libboost-all-dev

while in Fedora you can use

.. code-block:: bash

  sudo dnf install boost boost-devel

To install pytest you can use pip:

.. code-block:: bash

  pip install -U pytest

After having installed these libraries, make sure to configure with testing switched on:

.. code-block:: bash

  cmake .. -DENABLE_TESTING=ON

Now you can compile Tarantella and run its tests:

.. code-block:: bash

  make
  ctest
