## Configuring the Gitlab CI

### CI machines
* tnt_cluster (configuration [here]())
* vm_centos (configuration [here]())

Each machine is configured with one *gitlab-runner* that executes jobs sequentially. Jobs are tagged with the name of the specific machine they are defined for.

### GPU vs CPU jobs
*  GPU jobs will select only the tests requiring less than the maximum number of ranks specific to the test cluster (e.g., *tnt_cluster* has 4 nodes x 1 GPU per node).
*  CPU jobs can run multiple ranks per node, so that they will execute any number of ranks specified in the test definition.

### CI stages
* build

* cpp_tests
  Unit tests for the C++ libraries, which should be tested for different compilers/GPI-2 version.

* py_tests
  Python tests, which mainly depend on the TF version.

* integration tests
  Python tests, should be testing different TF versions.

### (Hidden) Cmake variables

* TNT_TEST_MACHINEFILE
Path to a specific CI machinefile.
Two predefined machinefiles exist for the *tnt_cluster*, one for GPUs (one rank per node) and one for CPUs (two ranks per node).



## Issues

### Compiler
* GPI-2 has to be compiled with the same compiler version as the code

### Tests fail with GCC-9.4.0/any Boost 0.67-0.69
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

### TF2.1
```
  File "/home/gitlab-runner/builds/yb9NsiCK/0/carpenamarie/ci_testing/src/examples/simple_FNN_GPI.py", line 158, in <module>
    tensorboard_callback],
  File "/home/gitlab-runner/builds/yb9NsiCK/0/carpenamarie/ci_testing/src/tarantella/model.py", line 113, in fit
    self._setup_for_execution('fit', x, y, callbacks, kwargs)
  File "/home/gitlab-runner/builds/yb9NsiCK/0/carpenamarie/ci_testing/src/tarantella/model.py", line 274, in _setup_for_execution
    self._preprocess_callbacks(callbacks)
  File "/home/gitlab-runner/builds/yb9NsiCK/0/carpenamarie/ci_testing/src/tarantella/model.py", line 332, in _preprocess_callbacks
    distributed_optimizer = self.dist_optimizer)
  File "/home/gitlab-runner/builds/yb9NsiCK/0/carpenamarie/ci_testing/src/tarantella/model.py", line 368, in __init__
    self._batches_seen_since_last_saving = keras_model_checkpoint._batches_seen_since_last_saving
AttributeError: 'ModelCheckpoint' object has no attribute '_batches_seen_since_last_saving'
```