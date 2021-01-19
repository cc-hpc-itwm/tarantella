Tarantella
============

Tarantella is an open-source, distributed Deep Learning framework built on top of TensorFlow 2,
providing scalable Deep Neural Network training on CPU and GPU compute clusters.

A comprehensive documentation of the Tarantella software can be found
[here](https://tarantella.readthedocs.io/en/latest/index.html).

## Prerequisites

### Compiler and build system

Tarantella can be built using a recent `gcc` compiler (starting from gcc-7.4.0).
`Cmake` (from version 3.8) is also required to build the library.

### SSH to localhost

To run GPI programs on the local machine, make sure you can ssh to localhost without password.

### Installing GPI-2

Compile and install the GPI-2 library (supported version: `v1.4.0`) with position independent flags (`-fPIC`).

```bash
git clone https://github.com/cc-hpc-itwm/GPI-2.git
git fetch --tags
git checkout -b v1.4.0 v1.4.0

./autogen.sh 
CFLAGS="-fPIC" CPPFLAGS="-fPIC" ./configure --with-ethernet --prefix=${YOUR_INSTALLATION_PATH}
make install

export PATH=${YOUR_INSTALLATION_PATH}/bin/:$PATH
```

### Create a Conda environment

We recommned `conda` to install Python packages. After installation, create and activate an environment:

```bash
conda create tarantella
conda activate tarantella
```
### Installing TensorFlow

`Tarantella` supports the following `TensorFlow` versions:
- Tensorflow 2.0
- Tensorflow 2.1
- Tensorflow 2.2

Either version can be installed in a conda environment using `pip`, as recommended
on the [TensorFlow website](https://www.tensorflow.org/install).

```bash
conda activate tarantella
conda install python=3.8
pip install --upgrade tensorflow==2.2
```

### Installing Pybind11

[Pybind11](https://github.com/pybind/pybind11) is available through `pip` and `conda`. However, the `pip`-package does not seem
to include a cmake package. This is why we recommend installing Pybind11 via `conda`.

Then you can install via

```bash
conda install pybind11 -c conda-forge
```

### (Optional) Installing Boost

To build `Tarantella` with tests, you will also need `boost`. In particular,
you will need the `devel`-packages.

In `Ubuntu` you can use
```bash
sudo apt install libboost-all-dev
```
while in `Fedora` use
```bash
sudo dnf install boost boost-devel
```

### (Optional) Installing pytest
Running the tarantella `python` tests requires [pytest](https://docs.pytest.org) packages.
```bash
pip install -U pytest
```

## Building and installing Tarantella

`Tarantella`'s build system uses `cmake`. For a standard out-of-source build (with optional tests), follow these steps:

```bash
git clone https://gitlab.itwm.fraunhofer.de/carpenamarie/hpdlf.git
cd hpdlf
mkdir build && cd build

export TARANTELLA_INSTALLATION_PATH=/your/installation/path
cmake -DCMAKE_INSTALL_PREFIX=${TARANTELLA_INSTALLATION_PATH} -DENABLE_TESTING=ON ..
make
ctest
```

Finally, install Tarantella to `TARANTELLA_INSTALLATION_PATH`:

```bash
make install
export PATH=${TARANTELLA_INSTALLATION_PATH}/bin:${PATH}

tarantella --version
```


## Distributed Training with Tarantella

There are two alternatives for running a model distributedly using Tarantella:
* Tarantella CLI: user-friendly and flexible
* direct execution through `gaspi_run`: for advanced users requiring customized runtime settings

### Using Tarantella CLI

A detailed description of all the command line options of `tarantella` can be found [here](https://tarantella.readthedocs.io/en/latest/quick_start.html#executing-your-model-with-tarantella).

The simplest way to train a model distributedly with Tarantella is to pass the Python script
to the ``tarantella`` command:
```bash
   tarantella -- model.py --batch_size=64 --learning_rate=0.01
```

This will execute our model distributedly on a single node, using all the available GPUs.
In case no GPUs can be found, ``tarantella`` will executed in serial mode on the CPU.
In case you have GPUs available, but want to execute ``tarantella`` on CPUs nonetheless,
you can specify the ``--no-gpu`` option.
```bash
   tarantella --no-gpu -- model.py
```

On a single node, we can also explicitly specify the number of TensorFlow instances
we want to use. This is done with the ``-n`` option, for example when training on 4 GPUs:
```bash
   tarantella -n 4 -- model.py --batch_size=64
```

Next, let's run ``tarantella`` on multiple nodes. In order to do this, we need to provide ``tarantella`` with a ``hostfile`` that contains the ``hostname`` s of the nodes that we want to use:
```bash
   $ cat hostfile
   name_of_node_1
   name_of_node_2
```

With this ``hostfile`` we can run ``tarantella`` on multiple nodes, using 2 GPUs on each node:
```bash
   tarantella --n-per-node=2 --hostfile hostfile -- model.py
```

### Run Tarantella models through `gaspi_run` directly

* Enable distributed training for an existing `Keras` model, by wrapping it as a `tarantella.Model`.
  A complete example can be found [here](https://gitlab.itwm.fraunhofer.de/carpenamarie/hpdlf/-/blob/master/src/examples/simple_FNN_GPI.py).

* Create a `nodesfile` with one machine name (i.e., hostname) per line for each of the parallel processes that will be involved in the data parallel DNN training.
  ```bash
  $cat ./nodesfile
  localhost
  localhost
  ```

* Create a `script.sh` file to execute a simple DNN training run:
  ```bash
  cat script.sh
  export PYTHONPATH=${SRC_PATH}/build/:${SRC_PATH}/src/
  python ${SRC_PATH}/src/examples/simple_FNN_GPI.py --batch_size=64 --number_epochs=1
  ```

  Depending on the specific configuration of the machine, the `script.sh` file may require more setup steps,
  such as activating the `conda` environment.
  Examples of additional environment configurations are shown below:
  ```bash
  export PATH=/path/to/gpi2/bin:/path/to/conda/environment:${PATH}
  export LD_LIBRARY_PATH=/path/to/conda/environment/lib:${LD_LIBRARY_PATH}

  conda activate tarantella
  ```

* Execute the script on all processes in parallel using:
  ```bash
  gaspi_run -m ./nodesfile -n ${NUM_PROCESSES} ./script.sh
  ```

  The `${NUM_PROCESSES}` variable should match the number of lines in the `nodesfile`.


## Benchmarks/Experiments

We provide a set of [Tarantella-enabled state-of-the-art 
models](https://gitlab.itwm.fraunhofer.de/carpenamarie/tnt_mlperf) adapted from the
[Tensorflow Model Garden](https://github.com/tensorflow/models).
To create reproducible experiments on HPC clusters, we used a dedicated [experiments engine](https://gitlab.itwm.fraunhofer.de/carpenamarie/exp_engine).

Experimental results can be found [here](https://gitlab.itwm.fraunhofer.de/labus/gpionnx_experiments).

## Troubleshooting

### SSH key configurations
* In order to run GPI programs, you need to be able to ssh to localhost without password. In order to do that
  ```bash
  cd ~/.ssh
  ssh-keygen
  ```
* Make sure not to overwrite existing keys.
Also take specific care that you set correct user rights on all files in `.ssh`
(cf. for instance [here](https://superuser.com/questions/215504/permissions-on-private-key-in-ssh-folder)).

* Append the public key to the authorized_keys file:
  ```bash
  cat id_rsa.pub >> authorized_keys
  ```

* Install and start an ssh server, e.g., openssh-server on Fedora.
More details [here](https://linuxconfig.org/how-to-install-start-and-connect-to-ssh-server-on-fedora-linux).


### Tarantella on the STYX GPU cluster

##### Installed Tarantella version

Tarantella `0.6.1` is already installed in STYX (in the *Tarantella* image).
It is compiled with `Tensorflow 2.2` and `Python 3.8.5`.

To use it, create a job using the *Tarantella* image and follow the steps below:

```bash
# load your own `conda` environment
conda activate my_env

# make sure Tensorflow 2.2 and Python 3.8.5 are installed
conda install python=3.8.5 tensorflow-gpu=2.2

# load tarantella environment
carme_prepare_tarantella
```

This is it, now you can run your code distributedly with (as shown [here](https://gitlab.itwm.fraunhofer.de/carpenamarie/hpdlf#distributed-training-with-tarantella)):
```bash
tarantella -- path/to/my/model.py
```

##### SSH configuration
* Use `hostname` instead of `localhost` for testing passwordless SSH access and for writing 
the `nodesfile` needed to execute GASPI-based code.

##### GPI-2 library

In case you want to run Tarantella directly using `gaspi_run`, the GPI-2 library is already
available in the *Base_image* and *Tarantella* images.

* `gaspi_run` can be used without any changes to the current ${PATH}
```bash
which gaspi_run
```

##### Installing TensorFlow with GPU support
* We have tested Tarantella with Tensorflow 2.1/2.2 installed from `conda` packages.
  ``` bash
  conda create -n tarantella_env
  conda activate tarantella_env
  conda install python=3.8 tensorflow-gpu=2.2
  conda install pybind11 -c conda-forge
  ```

##### Running MPI
* MPI is pre-installed on the GPU cluster (OpenMPI) and can be used for example for testing alternative frameworks
* To run MPI programs, replace calls to `mpirun` with `carme_mpirun`.

### Infiniband clusters

* Tarantella is compiled by default without Infiniband support. 
* Enable it before compiling the code, using the following `cmake` variable:
  ```bash
  cmake -DLINK_IB=ON ../
  ```

### Configuration on the Beehive cluster

to be filled

### FAQ
#### Execution error: `GPI library initialization incorrect environment vars`
* Make sure the code was executed through `gaspi_run` instead of being lunched directly using `python myfile.py`.

#### Execution error: `GPI library initialization general error`
* Occurs when the GPI library tries to connect to a previously used socket, which is not yet released.
* Fix: Retry running the code after a short wait until the port becomes available.

#### Execution hangs 
* Kill any processes that might be still running from a previous (failed) call to `gaspi_run`.

#### Build errors: Cmake cannot find Pybind11
* Error message
```
By not providing "Findpybind11.cmake" in CMAKE_MODULE_PATH this project has
asked CMake to find a package configuration file provided by "pybind11",
but CMake did not find one.

  Could not find a package configuration file provided by "pybind11" with any
  of the following names:

    pybind11Config.cmake
    pybind11-config.cmake

  Add the installation prefix of "pybind11" to CMAKE_PREFIX_PATH or set
  "pybind11_DIR" to a directory containing one of the above files.  If
  "pybind11" provides a separate development package or SDK, be sure it has
  been installed.
```
* Occurs when `pybind11` was installed using the `pip` package
* Fix: Install using `conda`, as recommended in the [installation guide](#installing-pybind11)

#### Build errors: Cmake does not detect the Python interpreter from the active Conda environment
* Manually add the path to the Conda environment `bin` directory to `PATH`
* Specify the path to the Python library from the command line when building the code
```bash
PATH_TO_CONDA_ENV=/path/to/conda/env
export PATH=${PATH_TO_CONDA_ENV}/bin:${PATH}
cmake -DPYTHON_EXECUTABLE=${PATH_TO_CONDA_ENV}/bin/python \
      -DPYTHON_LIBRARY=${PATH_TO_CONDA_ENV}/lib ../
```


