Advanced Topics
===============

This guide covers a number of advanced topics, such as
performance, reproducibility and user customization.

.. _ranks-label:

GASPI ranks
^^^^^^^^^^^


Using local batch sizes
^^^^^^^^^^^^^^^^^^^^^^^

As it has been stated in the :ref:`points to consider <points-to-consider-label>`, when using
Tarantella the user always specifies the *global* batch size. This has the advantage, that
the optimization process during the training of a DNN, and in particular the loss function do not
depend of the number of devices used during execution.

However, when the number of devices becomes
very large, the (device local) micro-batch size might become so small, that DNN kernel implementation
become less efficient and performance degradation takes place.
This is why it is in practice often advisable to scale the global batch size with the number of nodes.
This will often lead to linear speedups in terms of the time to accuracy when increasing
the number of devices used, at least up to some *critical batch size*, cf. [Shallue]_ and [McCandlish]_.
Changing the batch size of the optimizer, will however also imply the need to adapt the learning rate
schedule. For details, cf. for instance the :ref:`ResNet-50 tutorial <resnet50-label>`.

If you decide to scale the batch size with the number of nodes, Tarantella provides
two different ways to achieve this easily. The first option is, you multiply the local batch size
(for instance passed via a command line parameter) with the number of devices used,
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


.. _reproducibility-label:

Reproducibility
^^^^^^^^^^^^^^^

.. todo::

  * using ranks
  * setting local batch size
  * setting fusion threshold
  * reproducability (tf.random.set_seed, something with GPUs, something with datasets @Alex)

