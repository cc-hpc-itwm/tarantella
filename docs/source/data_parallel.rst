Distributed Data Parallel Training
==================================

The following section explains the parallelization strategy Tarantella uses to
provide distributed training. A full understanding thereof is, however, not required 
to be able to use the software. Please note the :ref:`points to consider <points-to-consider-label>`
to achieve best performance and reproducibility.

The general idea
----------------

In order to parallelize the training of DNNs, different, complementary strategies are available.
The conceptually simplest and most efficient one is called *data parallelism*. This strategy
is already in use when deploying batched optimizers, such as stochastic gradient descent (SGD)
or ADAM. In this case, input samples are grouped together in so-called mini-batches and
are processed in parallel.

Distribution of mini-batches
----------------------------

Tarantella extends this scheme by splitting each mini-batch into a number of micro-batches,
which are then executed on different devices (e.g., GPUs).
In order to do this, the DNN is replicated on each device,
which then processes part of the data independently of the other devices.
During the *backpropagation* pass, partial results need to be accumulated via a so-called
`allreduce <https://en.wikipedia.org/wiki/Collective_operation#All-Reduce_%5B5%5D>`_
collective operation.

Overlapping communication with computation
------------------------------------------

Tarantella implements this communication scheme using the
`Global Address Space Programming Interface (GASPI) <https://en.wikipedia.org/wiki/Global_Address_Space_Programming_Interface>`_.
This allows in particular to overlap the communication needed to execute *allreduce* operations
with the computation done in the *backpropagation* part of the DNN training.
This is done by starting *allreduce* operations as soon as the required local incoming gradients are
available, while continuing with *backpropagation* calculations at the same time.
The final, accumulated gradients are only expected once the entire *backpropagation* is completed.
This drastically mitigates the communication overhead introduced by the need to synchronize
the different devices, and leads to higher scalability.

.. _tensor-fusion-label:

Tensor Fusion
-------------

The granularity at which Tarantella executes *allreduce* operations can be varied from
one *allreduce* per layer (finest granularity) to one *allreduce* per iteration (coarsest granularity).
Using coarser granularities, i.e., *fusing* gradient tensors,
can lead to better bandwidth utilization, thus potentially increasing performance.
*Tensor Fusion* is set up before the first iteration of training and incurs no additional communication overhead.
Tarantella enables *Tensor Fusion* by default, but its granularity can be adjusted by the user,
cf. :ref:`here <tensor-fusion-threshold-label>`.

Model initialization and loading
--------------------------------

In order to guarantee that all devices have the same copy of the DNN when training is initiated,
the model needs to be communicated from one device to all the others.
This is done in Tarantella via the use of a so-called
`broadcast <https://en.wikipedia.org/wiki/Collective_operation#Broadcast_[3]>`_ operation.
This scheme applies both when the weights of a DNN are initialized randomly,
or loaded from a checkpoint.
As Tarantella provides this functionality automatically,
the user does not have to take care of it.

.. _points-to-consider-label:

Distributed Datasets
=====================

In order to process micro-batches independently on each device and to obtain the same results
as in serial execution, the input data of each mini-batch has to be split and distributed
among all devices.

Tarantella automatically takes care of this through the use of distributed datasets.
The user simply provides Tarantella with a ``tf.data.Dataset`` that is batched
with the mini-batch size. Tarantella will then automatically distribute the input data
by sharding the mini-batch into individual micro-batches. Sharding is done at the level
of samples (as opposed to e.g., files) to ensure :ref:`reproducibility <reproducibility-label>`
of serial results.

To guarantee reproducibility, it is also important that shuffling of samples is done
in the same way on all devices. Tarantella does this using either the ``seed`` provided
by the user, or a specific default seed. Please refer to the
:ref:`Quick Start <using-distributed-datasets-label>`
for more details.

Points to Consider
==================

.. _global-vs-local-batch-size-label:

Global versus local batch size
------------------------------

As explained above, when using data parallelism, there exists a *mini-batch size*
(in the following also called global batch size or simply batch size) 
as well as a *micro-batch size* (also called local batch size).
The former represents the number of samples that
is averaged over in the loss function of the optimizer, and is equivalent to
the (mini-)batch size used in non-distributed training. The latter is the number
of samples that is processed locally by each of the devices per iteration.

.. note::

   In Tarantella, the user always specifies the **global batch size**.

Using a strictly synchronous optimization scheme, and by carefully handling the data distribution,
**Tarantella guarantees the reproducibility of DNN training results independently of the number of
devices used**, as long as all hyperparameters (such as global batch size and learning rate)
are kept constant. [#footnote_random_seeds]_

However, to achieve best performance for certain DNN operators (`Conv2D`, `Dense`, etc.)
it is often advisable to *keep the local batch size constant*, and scale the global
batch size with the number of devices used. This, in turn, will force you to
adjust other hyperparameters, such as the learning rate, in order to converge
to a comparable test accuracy, as observed for instance in [Shallue]_.

In practice, the use of a learning rate schedule with initial *warm up* and
a *linear learning rate scaling* [Goyal]_, as it is described
:ref:`here <resnet50-label>`, often suffices. 

.. tip::

   For best performance, scale the batch size with the number of devices used,
   and :ref:`adapt the learning rate schedule <resnet50-label>`.

Batch normalization layers
--------------------------

The issue of global versus local batch size particularly affects the layers
that calculate (and learn) statistics over entire batches.
A well-known example of this type of layer is
`batch normalization <https://en.wikipedia.org/wiki/Batch_normalization>`_.

.. caution::

   Tarantella always calculates batch statistics over **local batches**.

As a consequence, the training results for DNNs with batch-normalization layers
**will not be identical when changing the number of devices, even if
the global batch size stays the same.**
At the moment, this can be circumvented by using normalization layers that
do *not* average over entire batches, such as instance normalization
[Ulyanov]_.

Averaging over *local* batches instead of global batches should in practice
have only minor influence on the quality of the final test accuracy.
Note however, the extreme case of very small *local* batch sizes.

.. caution::

   Avoid using ``BatchNormalization`` layers when the global batch size
   divided by the number of devices used is *smaller than 16*.

In such cases, the local batches that are used to collect statistics are
too small to obtain meaningful results. This will likely reduce the
benefits of batch normalization, cf. for instance [Yang]_ and [Uppal]_.
In this case, please consider increasing the global batch size,
or reducing the number of devices used.

Managing individual devices
---------------------------

Although Tarantella's user interface abstracts away most of the details of
parallel programming, it is sometimes useful to be able to control
Python code execution at device level. This can be achieved using the
`GASPI <https://en.wikipedia.org/wiki/Global_Address_Space_Programming_Interface>`_ concept
of a ``rank``. Details on how to do this can be found in the
:ref:`advanced topics <ranks-label>`.

.. rubric:: References

.. [Shallue] Shallue, Christopher J., et al. "Measuring the effects of data parallelism on neural network training." arXiv preprint arXiv:1811.03600 (2018).

.. [Ulyanov] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).

.. [Goyal] Goyal, Priya, et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv preprint arXiv:1706.02677 (2017).

.. [Yang] Yang, Greg, et al. "A mean field theory of batch normalization." arXiv preprint arXiv:1902.08129 (2019).

.. [Uppal] https://towardsdatascience.com/curse-of-batch-normalization-8e6dd20bc304

.. [McCandlish] McCandlish, Sam, et al. "An empirical model of large-batch training." arXiv preprint arXiv:1812.06162 (2018).

.. [He] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [Vaswani] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

.. rubric:: Footnotes

.. [#footnote_random_seeds] This is strictly true, only when all randomness in TensorFlow is
   seeded or switched off, as explained in the :ref:`advanced topics <reproducibility-label>`

