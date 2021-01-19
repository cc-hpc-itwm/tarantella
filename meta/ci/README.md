## Configuring the Gitlab CI

### CI machines
* tnt_cluster (configuration [here](https://gitlab.itwm.fraunhofer.de/carpenamarie/hpdlf/-/blob/master/meta/ci/tnt_cluster.yml))
* vm_centos (configuration [here](https://gitlab.itwm.fraunhofer.de/carpenamarie/hpdlf/-/blob/master/meta/ci/vm_centos.yml))

Each machine is configured with one *gitlab-runner* that executes jobs sequentially. Jobs are tagged with the name of the specific machine they are defined for.

### GPU vs CPU jobs
*  GPU jobs will select only the tests requiring less than the maximum number of ranks specific to the test cluster (e.g., *tnt_cluster* has 4 nodes x 1 GPU per node).
*  CPU jobs can run multiple ranks per node, so that they will execute any number of ranks specified in the test definition.

### CI stages
* build

* unit_tests_cpp
  Unit tests for the C++ libraries, which should be tested for different compilers/GPI-2 version.

* unit_tests_python
  Python unit tests, which mainly depend on the TF version and do not require DNN training (e.g., distributed datasets test).

* integration tests
  Python tests, should be executed for different TF versions. 

### (Hidden) Cmake variables

* TNT_TEST_MACHINEFILE
Path to a specific CI machinefile.
Two predefined machinefiles exist for the *tnt_cluster*, one for GPUs (one rank per node) and one for CPUs (two ranks per node).



## Issues

### Compiler
* GPI-2 has to be compiled with the same compiler version as the code

### Tests fail with GCC-9.4.0/any Boost 0.67-0.69
* compiles
* test run correctly with "Debug" build
* with "Release" build:
```
Startup time: 0 sec
*** buffer overflow detected ***: terminated
unknown location(0): fatal error: in "Test setup": signal: SIGABRT (application abort requested)
Running 12 test cases...
```

### Boost 0.70
* cannot compile Boost tests


### TF2.2 with GPUs
* the `tensorflow-gpu=2.2` from conda does not actually use the GPUs
