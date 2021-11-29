![Tarantella](docs/source/pics/tnt_logo_text.png)

<br/><br/>

Tarantella is an open-source, distributed Deep Learning framework built on top of TensorFlow,
providing scalable Deep Neural Network training on CPU and GPU compute clusters.

Tarantella is easy-to-use, allows to re-use existing TensorFlow models,
and does not require any knowledge of parallel computing.


## Goals

Tarantella is designed to meet the following goals:

* ease of use
* synchronous training scheme
* seamless integration with existing Keras models
* support for GPU and CPU systems
* strong scalability

## Install

To build Tarantella from source, the following dependencies are required:

* [TensorFlow](https://www.tensorflow.org/install) (supported versions TensorFLow 2.0-2.7)
* [GaspiCxx](https://github.com/cc-hpc-itwm/GaspiCxx) (version 1.0)
* [GPI-2](https://github.com/cc-hpc-itwm/GPI-2) (from version 1.4.0)
* [pybind11](https://github.com/pybind/pybind11) (from version 2.4.3)
* Python (from version 3.7)
* C++ compiler (e.g., `gcc` from version 7.4.0)
* CMake (from version 3.8)

Detailed installation instructions can be found in the [technical docs](https://tarantella.readthedocs.io/en/latest/installation.html).

## Resources

* [Official website](https://www.tarantella.org)
* [Technical documentation](https://tarantella.readthedocs.io/en/latest)

## License

[License](LICENSE)
