Advanced Topics
===============

This guide covers a number of advanced topics, such as
performance, reproducibility and user customization.


.. _ranks-label:

GASPI ranks
^^^^^^^^^^^

In order to execute distributed DNN training, Tarantella starts multiple processes
on different devices. These processes will be assigned different IDs by the GASPI
communication library, in order to organize communication and synchronization between
the different devices. These IDs are called *ranks*. Usually, Tarantella abstracts away
the concept of *ranks*, in such a way that Tarantella's user interface is essentially
the same as Keras' user interface.

However, sometimes it is useful, to execute a specific part of code only on one
or a subgroup of all ranks. In particular, one sometimes wants to execute a code
block on the devices that started ``tarantella``, the so-called *master rank*.

To access ranks, Tarantella provides the following functions

* ``tnt.get_rank()``
* ``tnt.get_size()``
* ``tnt.get_master_rank()``
* ``tnt.is_master_rank()``

``tnt.get_rank()`` returns the ID of the local rank.
``tnt.get_size()`` returns the total number of ranks.
``tnt.get_master_rank()`` and ``tnt.is_master_rank()`` return the ID of the master rank
and a boolean for whether the local rank is the master rank or not, respectively.

Here is a simple example, when using the master rank can be useful to print notifications
only once to ``stdout``:

.. code-block:: python

   if tnt.is_master_rank():
     print("Printing from the master rank")

In the same vein, you might want to use ranks to execute :ref:`callbacks <callbacks-label>` for logging 
only on one rank:

.. code-block:: python

   history_callback = tf.keras.callbacks.History()
   tnt_model.fit(train_dataset,
                 callbacks = [history_callback] if tnt.is_master_rank() else [])


.. _using-local-batch-sizes-label:

Using local batch sizes
^^^^^^^^^^^^^^^^^^^^^^^

As it has been stated in the :ref:`points to consider <points-to-consider-label>`, when using
Tarantella the user always specifies the *global* batch size. This has the advantage that
the optimization process during the training of a DNN, and in particular the loss function do not
depend on the number of devices used during execution.

However, when the number of devices becomes
very large, the (device-local) micro-batch size might become so small, that DNN kernel implementations
are less efficient, resulting in overall performance degradation.
This is why it is in practice often advisable to scale the global batch size with the number of nodes.
This will often lead to linear speedups in terms of the time to accuracy when increasing
the number of devices used, at least up to some *critical batch size*, cf. [Shallue]_ and [McCandlish]_.
Changing the batch size of the optimizer will however also imply the need to adapt the learning rate
schedule.

.. todo::
  
  Enable when the Tutorial is updated:
  For details, cf. for instance the :ref:`ResNet-50 tutorial <resnet50-label>`.

If you decide to scale the batch size with the number of nodes, Tarantella provides
two different ways to achieve this easily. The first option is to multiply the local batch size
(for instance passed via a command-line parameter) with the number of devices used,
batch your dataset with it, and call ``fit`` on it:

.. code-block:: python

   micro_batch_size = args.micro_batch_size
   batch_size = tnt.get_size() * micro_batch_size
   train_dataset = train_dataset.batch(batch_size)
   tnt_model.fit(train_dataset)

As a second option you can also pass the local batch size directly to the ``tnt_micro_batch_size``
parameter in fit, and leave your dataset unbatched:

.. code-block:: python

   micro_batch_size = args.micro_batch_size
   tnt_model.fit(train_dataset,
                 tnt_micro_batch_size = micro_batch_size)

This parameter is also available in ``evaluate`` and ``predict``. In addition, ``fit`` also supports
setting the validation set micro batch size in a similar way with ``tnt_validation_micro_batch_size``.
For more information, please also read :ref:`using distributed datasets <using-distributed-datasets-label>`.


.. _tensor-fusion-threshold-label:

Setting Tensor Fusion threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tarantella automatically uses :ref:`Tensor Fusion <tensor-fusion-label>` with a default
threshold of 32kB. This threshold specifies the minimal size of local buffers in *allreduce*
communication operations used to accumulate partial gradients during *backpropagation*.

Note that the threshold value implies a trade-off between the potential to utilize network
bandwidth, and the overlap of computation and communication during *backpropagation*. The
larger the threshold, the more bandwidth-bound the *allreduce* algorithm will get, but
the less potential there will be to overlap its execution with kernel computations.
Also note that the ideal threshold value will generally depend on the number of nodes used.

To change the default value, you can pass a threshold value in kB to ``tarantella``:

.. code-block:: bash

   tarantella --hostfile hostfile --fusion-threshold=<FUSION_THRESHOLD_KB> -- model.py


.. _reproducibility-label:

Reproducibility
^^^^^^^^^^^^^^^

Reproducibility is a very important prerequisite to obtain meaningful results in
scientific computing and research. Unfortunately, using stochastic algorithms,
pseudo random generators and having to deal with the pitfalls of floating-point arithmetics,
it is particularly difficult to achieve reproducibility in Deep Learning research.

In order to be able to reproduce results obtained with TensorFlow, when running in
a multi-node/multi-device setting with Tarantella, one needs to meet at least 
the following requirements:

* set the random seed with ``tf.random.set_seed(seed)``
* set the environment variable ``os.environ['TF_CUDNN_DETERMINISTIC']='1'``
* set the shuffle seeds when using ``tf.data.Dataset`` with ``shuffle(seed=seed)`` and ``list_files(seed=seed)``
* set the ``deterministic`` parameter to ``True`` in ``Dataset`` transformations such as ``interleave`` and ``map``
* make sure the number of samples in your datasets equal a multiple of ``batch_size``
