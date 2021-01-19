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

### TF2.2 with GPUs

```
31: =================================== FAILURES ===================================
31: _ TestsDataParallelOptimizers.test_compare_accuracy_optimizers[230-64-RMSprop-lenet5_model_generator] _
31: 
31: self = <optimizers_mnist_test.TestsDataParallelOptimizers object at 0x7f36e98501c0>
31: tarantella_framework = <module 'tarantella' from '/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/src/tarantella/__init__.py'>
31: keras_model = <function lenet5_model_generator at 0x7f36e7f55b80>
31: optimizer = <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>
31: micro_batch_size = 64, nbatches = 230
31: 
3
31:         model.compile(optimizer(learning_rate=lr),
31:                       loss = keras.losses.SparseCategoricalCrossentropy(),
31:                       metrics = [keras.metrics.SparseCategoricalAccuracy()])
31: >       model.fit(train_dataset,
31:                   epochs = number_epochs,
31:                   verbose = 0)
31: 
31: builds/1xsBxmkN/0/carpenamarie/hpdlf/test/python/data_parallel_training/optimizers_mnist_test.py:39: 
31: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
31: builds/1xsBxmkN/0/carpenamarie/hpdlf/src/tarantella/model.py:142: in fit
31:     return self.model.fit(x,
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:66: in _method_wrapper
31:     return method(self, *args, **kwargs)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:848: in fit
31:     tmp_logs = train_function(iterator)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py:580: in __call__
31:     result = self._call(*args, **kwds)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py:627: in _call
31:     self._initialize(args, kwds, add_initializers_to=initializers)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py:505: in _initialize
31:     self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/function.py:2446: in _get_concrete_function_internal_garbage_collected
31:     graph_function, _, _ = self._maybe_define_function(args, kwargs)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/function.py:2777: in _maybe_define_function
31:     graph_function = self._create_graph_function(args, kwargs)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/function.py:2657: in _create_graph_function
31:     func_graph_module.func_graph_from_py_func(
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py:981: in func_graph_from_py_func
31:     func_outputs = python_func(*func_args, **func_kwargs)
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py:441: in wrapped_fn
31:     return weak_wrapped_fn().__wrapped__(*args, **kwds)
31: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
31: 
31: args = (<tensorflow.python.data.ops.iterator_ops.OwnedIterator object at 0x7f36d6b9d610>,)
31: kwargs = {}
31: 
31:     def wrapper(*args, **kwargs):
31:       """Calls a converted version of original_func."""
31:       # TODO(mdan): Push this block higher in tf.function's call stack.
31:       try:
31:         return autograph.converted_call(
31:             original_func,
31:             args,
31:             kwargs,
31:             options=autograph.ConversionOptions(
31:                 recursive=True,
31:                 optional_features=autograph_options,
31:                 user_requested=True,
31:             ))
31:       except Exception as e:  # pylint:disable=broad-except
31:         if hasattr(e, "ag_error_metadata"):
31: >         raise e.ag_error_metadata.to_exception(e)
31: E         RuntimeError: in user code:
31: E         
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:571 train_function  *
31: E                 outputs = self.distribute_strategy.run(
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:951 run  **
31: E                 return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2290 call_for_each_replica
31: E                 return self._call_for_each_replica(fn, args, kwargs)
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2649 _call_for_each_replica
31: E                 return fn(*args, **kwargs)
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:540 train_step  **
31: E                 _minimize(self.distribute_strategy, tape, self.optimizer, loss,
31: E             /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1803 _minimize
31: E                 gradients = optimizer._aggregate_gradients(zip(gradients,  # pylint: disable=protected-access
31: E             /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/src/tarantella/optimizers/synchronous_distributed_optimizer.py:28 _aggregate_gradients
31: E                 self.comm.setup_infrastructure(grads_and_vars)
31: E             /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/src/tarantella/__init__.py:142 setup_infrastructure
31: E                 self.comm = GPICommLib.SynchDistCommunicator(global_context, grad_infos, self.threshold)
31: E         
31: E             RuntimeError: Context::allocate_segment : segment could not be allocatedDevice operation error
31: 
31: /opt/environments/tf2.2/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py:968: RuntimeError
31: ----------------------------- Captured stderr call -----------------------------
31: [Rank    1]: Error 12 (Cannot allocate memory) at (devices/ib/GPI2_IB_SEG.c:45):Memory registration failed (libibverbs)
31: ------------------------------ Captured log call -------------------------------
31: INFO     root:mnist_models.py:68 Initialized LeNet5 model
31: ---------------------------- Captured log teardown -----------------------------
31: INFO     root:conftest.py:16 teardown tarantella
31: =========================== short test summary info ============================
31: FAILED builds/1xsBxmkN/0/carpenamarie/hpdlf/test/python/data_parallel_training/optimizers_mnist_test.

31: ========================= 1 failed, 2 passed in 18.86s =========================
31: corrupted size vs. prev_size while consolidating
31: Fatal Python error: Aborted
31: 
31: Current thread 0x00007fe807665740 (most recent call first):
31: <no Python frame>
31: corrupted size vs. prev_size while consolidating
31: Fatal Python error: Aborted
31: 
31: Current thread 0x00007f0170686740 (most recent call first):
31: <no Python frame>
31: double free or corruption (top)
31: Fatal Python error: Aborted
31: 
31: Current thread 0x00007f374e1e7740 (most recent call first):
31: <no Python frame>
31: double free or corruption (top)
31: Fatal Python error: Aborted
31: 
31: Current thread 0x00007f21a6971740 (most recent call first):
31: <no Python frame>
31: /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/build/test/python/run_OptimizersDataParallelMNIST.sh: line 7: 482172 Aborted                 (core dumped) /opt/environments/tf2.2/bin/python -m pytest /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/test/python/data_parallel_training/optimizers_mnist_test.py
31: /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/build/test/python/run_OptimizersDataParallelMNIST.sh: line 7: 694808 Aborted                 (core dumped) /opt/environments/tf2.2/bin/python -m pytest /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/test/python/data_parallel_training/optimizers_mnist_test.py
31: /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/build/test/python/run_OptimizersDataParallelMNIST.sh: line 7: 2924610 Aborted                 (core dumped) /opt/environments/tf2.2/bin/python -m pytest /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/test/python/data_parallel_training/optimizers_mnist_test.py
31: Aborted (core dumped)
31: Sleep 2
31: Killing 0 processes
31: 'sh' '-c' 'ps -ef | grep "/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/build" | grep -v grep | grep -v "/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/build/test/python/run_OptimizersDataParallelMNIST.sh" | awk '{print $2}' | xargs -r kill -9 || 1'
31: '/opt/tools/spack/opt/spack/linux-ubuntu20.04-broadwell/gcc-8.4.0/gpi-2-1.4.0-vnguxg5rasd3pg5j3js5u6cb755bvgzv/bin/gaspi_run' '-n' '4' '-m' '/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/meta/ci/tnt_cluster/machinefile_gpu' '/bin/bash' '/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/cmake/cleanup.sh'
31: Startup time: 0 sec
31: CMake Error at /home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/cmake/run_test.cmake:52 (message):
31:   Test failed:'1'
31: 
31: 
1/2 Test #31: OptimizersDataParallelMNIST_4ranks ...***Failed   24.84 sec
test 1
    Start  1: gpi_cleanup

1: Test command: /usr/bin/sh "/home/gitlab-runner/builds/1xsBxmkN/0/carpenamarie/hpdlf/cmake/cleanup.sh"
1: Test timeout computed to be: 10000000
2/2 Test  #1: gpi_cleanup ..........................   Passed    0.03 sec

The following tests passed:
	gpi_cleanup

```
