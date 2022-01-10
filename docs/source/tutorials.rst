.. _tutorials-label:

Tutorials
=========

This section delves into more advanced usage of Tarantella with the help of
state-of-the-art models for two widely-used applications in Deep Learning:

* Image classification: ResNet-50
* Machine translation: Transformer

The models shown here are adapted from the
`TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/official>`_.
While the model implementations and hyperparameters are unchanged to preserve
compatibility with the TensorFlow official models, we provide simplified training
schemes that allow for a seamless transition from basic serial training to distributed
data parallelism using Tarantella.


Prerequisites
-------------

The tutorial models can be downloaded from the
`Tnt Models repository <https://github.com/cc-hpc-itwm/tarantella_models>`_.

.. code-block:: bash

  cd /your/models/path
  git clone https://github.com/cc-hpc-itwm/tarantella_models

  cd tarantella_models/src
  export TNT_MODELS_PATH=`pwd`

To use these models, install the the following dependencies:

* TensorFlow 2.7
* Tarantella 0.7.1

For a step-by-step installation, follow the :ref:`installation-label` guide.
In the following we will assume that TensorFlow was installed in a ``conda`` 
environment called ``tarantella``.

Now we can install the final dependency,
`TensorFlow official Model Garden <https://github.com/tensorflow/models>`__:

.. code-block:: bash

    conda activate tarantella
    pip install tf-models-official==2.7


.. _resnet50-label:

ResNet-50
---------

Deep Residual Networks (ResNets) represented a breakthrough in the field of
computer vision, enabling deeper and more complex deep convolutional networks.
Introduced in [He]_, ResNet-50 has become a standard model for image classification
tasks, and has been shown to scale to very large number of nodes in data parallel
training [Goyal]_.

Run Resnet-50 with Tarantella
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before running the model, we need to add it to the existing ``PYTHONPATH``.

.. code-block:: bash

    export PYTHONPATH=${TNT_MODELS_PATH}/models/resnet:${PYTHONPATH}

Furthermore, the ``ImageNet`` dataset needs to be installed and available on
all the nodes that we want to use for training.
TensorFlow provides convenience scripts to download datasets, in their ``datasets``
package that is installed as a dependency for the TensorFlow Model Garden.
Install ImageNet to your local machine as described
`here <https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/scripts/download_and_prepare.py>`_.

.. code-block:: bash

    export TNT_DATASETS_PATH=/path/to/downloaded/datasets

    python -m tensorflow_datasets.scripts.download_and_prepare \
    --datasets=imagenet2012 --data_dir=${TNT_DATASETS_PATH}


Let's assume we have access to two nodes (saved in ``hostfile``) equipped with 4 GPUs each.
We can now simply run the ResNet-50 as follows:

.. code-block:: bash

    tarantella --hostfile ./hostfile --devices-per-node 4 \
    -- ${TNT_MODELS_PATH}/models/resnet/resnet50_tnt.py --data_dir=${TNT_DATASETS_PATH} \
                                                        --batch_size=512 \
                                                        --train_epochs=90 \
                                                        --epochs_between_evals=10

The above command will train a ResNet-50 models on the 8 devices available in parallel
for ``90`` epochs, as suggested in [Goyal]_ to achieve convergence.
The ``--epochs_between_evals`` parameter specifies the frequency of evaluations of the
*validation dataset* performed in between training epochs.

Note the ``--batch_size`` parameter, which specifies the global batch size used in training.

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^
We will now look closer into the implementation of the ResNet-50 training scheme.
The main training steps reside in the ``models/resnet/resnet50_tnt.py`` file.

The most important step in enabling data parallelism with Tarantella is
to wrap the Keras model:

.. code-block:: python

    model = resnet_model.resnet50(num_classes = imagenet_preprocessing.NUM_CLASSES)
    model = tnt.Model(model)

Next, the training procedure can simply be written down as it would be for a
standard, TensorFlow-only model. No further changes are required to train the
model in a distributed manner.

In particular, the ImageNet dataset is loaded and preprocessed as follows:

.. code-block:: python

    train_dataset = imagenet_preprocessing.input_fn(is_training = True,
                                                    data_dir = flags_obj.data_dir,
                                                    batch_size = flags_obj.batch_size,
                                                    shuffle_seed = 42,
                                                    drop_remainder = True)

The
`imagenet_preprocessing.input_fn
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/resnet/imagenet_preprocessing.py#L31>`_
function reads the input files in ``data_dir``, loads the training samples, and processes
them into TensorFlow datasets.

The user only needs to pass the global ``batch_size`` value, and the Tarantella
framework will ensure that the dataset is properly distributed among devices,
such that:

  * each device will process an independent set of samples
  * each device will group the samples into micro batches, where the micro-batch
    size will be roughly equal to ``batch_size / num_devices``.
    If the batch size is not a multiple of the number of ranks, the remainder samples
    will be equally distributed among the participating ranks, such that some ranks
    will use a micro-batch of ``(batch_size / num_devices) + 1``.
  * each device will apply the same set of transformations to its input samples as
    specified in the ``input_fn`` function.

The advantage of using the *automatic dataset distribution* mechanism of Tarantella
is that users can reason about their I/O pipeline without taking care of the details
about how to distribute it.
To disable the *automatic dataset distribution* and ensure each rank will use the
same micro-batch size equal to ``batch_size / num_devices``, run the code with the
following flag: ``--auto_distributed=False``. Make sure the provided ``batch_size``
is a multiple of the number of ranks in this case.

Before starting the training, the model is compiled using a standard Keras optimizer
and loss.

.. code-block:: python

    model.compile(optimizer = optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = (['sparse_categorical_accuracy']))

We provide flags to enable the most commonly used Keras ``callbacks``, such as
the ``TensorBoard`` profiler, which can simply be passed to the ``fit`` function
of the Tarantella model.

.. code-block:: python

    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir = flags_obj.model_dir,
                                                    profile_batch = 2))

If model checkpointing is required, it can be enabled through the ``ModelCheckpoint``
callback as usual (cf. :ref:`checkpointing models with Tarantella <checkpointing-via-callbacks-label>`).

.. code-block:: python

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))


There is no need for any further changes to proceed with distributed training:

.. code-block:: python

    history = model.fit(train_dataset,
                        epochs = flags_obj.train_epochs,
                        callbacks = callbacks,
                        validation_data = validation_dataset,
                        validation_freq = flags_obj.epochs_between_evals,
                        verbose = 1)


Advanced topics
^^^^^^^^^^^^^^^

Scaling the batch size
""""""""""""""""""""""

Increasing the batch size provides a simple means to achieve significant training
time speed-ups, as it leads to perfect scaling with respect to the steps required
to achieve the target accuracy (up to some dataset- and model- dependent critical
size, after which further increasing the batch size only leads to diminishing returns)
[Shallue]_.

This observation, together with the fact that small local batch sizes decrease the
efficiency of DNN operators, represent the basis for a standard technique in data
parallelism: *using a fixed micro batch size and scaling the global batch size
with the number of devices*.

Tarantella provides multiple mechanisms to set the batch size, as presented in the
:ref:`Quick Start guide<using-distributed-datasets-label>`.

In the case of ResNet-50, we specify the global batch size as a command line
parameter, and let the framework divide the dataset into microbatches.

.. _scale-learning-rate-label:

Scaling the learning rate
"""""""""""""""""""""""""

To be able to reach the same target accuracy when scaling the global batch size up,
other hyperparameters need to be carefully tuned [Shallue]_.
In particular, adjusting the learning rate is essential for achieving convergence
at large batch sizes. [Goyal]_ proposes to *scale the
learning rate up linearly with the batch size* (and thus with the number of devices).

The scaled-up learning rate is set up at the begining of training, after which the
learning rate evolves over the training steps based on a so-called
*learning rate schedule*.

In our ResNet-50 example, we use the
`PiecewiseConstantDecayWithWarmup <https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/resnet/resnet50_tnt.py#L20>`__
schedule provided by the TensorFlow Models implementation, which is similar to the schedule
introduced by [Goyal]_.
When training starts, the learning rate is initialized to
a large value that allows to explore more of the search space. The learning rate will
then monotonically decay the closer the algorithm gets to convergence.

The initial learning rate here is scaled up by a factor computed as:

.. code-block:: bash

  self.rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size

Here ``batch_size`` is the global batch size and ``base_lr_batch_size`` is the predefined batch size
(set to ``256``) that corresponds to single-device training. This effectively scales the
``BASE_LEARNING_RATE`` linearly with the number of devices used.

Learning rate warm-up
"""""""""""""""""""""

Whereas scaling up the learning rate with the batch size is necessary, a large learning
rate might degrade the stability of the optimization algorithm, especially in early training.
A technique to mitigate this limitation is to *warm-up* the learning rate during the first
epochs, particularly when using large batches [Goyal]_.

In our ResNet-50 example, the `PiecewiseConstantDecayWithWarmup` schedule
starts with a small value for the learning rate, which then increases at every step
(i.e., iteration), for a number of initial
`warmup_steps <https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/resnet/common.py#L30>`_.

The ``warmup_steps`` value defaults to the number of iterations of the first five epochs,
matching the schedule proposed by [Goyal]_.
After the ``warmup_steps`` are done, the learning rate value should reach the *scaled initial
learning rate* introduced above.

.. code-block:: python

  def warmup_lr(step):
    return self.rescaled_lr * (
        tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))

.. _transformer-label:

Transformers
------------

The Transformer is a Deep Neural Network widely used in the field of natural language
processing (NLP), in particular for tasks such as machine translation.
It was first proposed by [Vaswani]_.

Run the Transformer with Tarantella
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Tranformer training scheme can be found
`here <https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/transformer_tnt.py>`__,
and has to be added to 
the existing ``PYTHONPATH``:

.. code-block:: bash

    export PYTHONPATH=${TNT_MODELS_PATH}/models/transformer:${PYTHONPATH}

We will follow the training procedure presented in [Vaswani]_, where the authors 
show results for training the `big` variant of the Transformer model on 
a machine translation dataset called 
`WMT14 <http://www.statmt.org/wmt14/translation-task.html>`_.

To install the dataset, we will use the Tensorflow ``datasets`` package, which 
should have been already installed in your ``conda`` environment as a 
dependency for the TensorFlow Model Garden, and download the English-German 
dataset to match the results by [Vaswani]_.
Detailed instructions on how to obtain the dataset are provided in the 
`TensorFlow documentation <https://www.tensorflow.org/datasets/catalog/wmt14_translate>`_.

Now we can start training.
Once again, let's assume we have access to two nodes (specified in ``hostfile``)
equipped with 4 GPUs each.

.. code-block:: bash

    export WMT14_PATH=/path/to/the/installed/dataset

    tarantella --hostfile ./hostfile --devices-per-node 4 \
    -- ${TNT_MODELS_PATH}/models/transformer/transformer_tnt.py \
                         --data_dir=${WMT14_PATH} \
                         --vocab_file=${WMT14_PATH}/vocab.ende.32768 \
                         --bleu_ref=${WMT14_PATH}/newstest2014.de \
                         --bleu_source=${WMT14_PATH}/newstest2014.en \
                         --param_set=big \
                         --train_epochs=30 \
                         --epochs_between_evals=30 \
                         --batch_size=32736

The above command will select the ``big`` model implementation and train it
on the 8 specified devices in a distributed fashion.
To reach the target accuracy, [Vaswani]_ specifies that the model needs to be 
trained for ``30`` epochs.

The Transformer requires access to a vocabulary file, which contains all the
tokens derived from the dataset. This is provided as the ``vocab_file`` parameter
and is part of the pre-processed dataset.

After training, one round of evaluation is conducted using the ``newstest2014``
dataset to translate English sentences into German. The frequency of evaluation
rounds can be changed by updating the `epochs_between_evals` parameter.

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^

The Transformer model itself is implemented and imported from the 
`TensorFlow Model Garden 
<https://github.com/tensorflow/models/tree/master/official/nlp/transformer>`__.
The training procedure and dataset loading and pre-processing do not require
extensive changes to work with Tarantella. However, we provide a simplified 
version to highlight the usage of Tarantella with Keras training loops.

Thus, the Keras transformer model is created in
`TransformerTntTask class
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/transformer_tnt.py#L80>`_.
Two different versions of the model are used, one for training (wrapped into
a Tarantella model), and one for inference (serial Keras model).

.. code-block:: python

  self.train_model = create_model(internal_model, self.params, is_train = True)
  # Enable distributed training
  self.train_model = tnt.Model(self.train_model)

  # The inference model is wrapped as a different Keras model that does not use labels
  self.predict_model = create_model(internal_model, self.params, is_train = False)

To illustrate alternatives in the use of Tarantella, we distribute the data
manually here, `data_pipeline.py
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/data_pipeline.py>`_
file, as explained in the
:ref:`manually-distributed datasets<manually-distributed-datasets-label>` section.
Alternatively, automatic dataset distribution could be used, as explained in the
:ref:`Quick Start<using-distributed-datasets-label>`.

To be able to manually split the dataset across ranks, we need access to **rank IDs**
and the **total number of ranks**, which are then passed to the `IO pipeline
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/transformer_tnt.py#L134>`_.

The :ref:`Advanced Topics<ranks-label>` section explains the API Tarantella
exposes to access ranks.

.. code-block:: python

  train_ds = data_pipeline.train_input_fn(self.params,
                                          shuffle_seed = 42,
                                          num_ranks = tnt.get_size(),
                                          rank = tnt.get_rank())


Here, the ``data_pipeline.train_input_fn`` reads in the dataset and applies a series 
of transformations to convert it into a batched set of sentences.

Next, the user can also create callbacks, which can then be simply passed on to
the training function.

.. code-block:: python

  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.flags_obj.model_dir))

Finally, we can call ``model.fit`` to start distributed training on all devices:

.. code-block:: python

    history = model.fit(train_ds,
                        tnt_distribute_dataset = False,
                        epochs=self.params["train_epochs"],
                        callbacks=callbacks,
                        verbose=1)

In the following sections we will show how we modify the ``fit`` loop to allow for
a customized evaluation of the trained model.

Important points
^^^^^^^^^^^^^^^^

Customized behavior based on **rank**
"""""""""""""""""""""""""""""""""""""

Although all ranks participating in data parallel training use identical replicas
of the same model and make progress in sync, there are cases when certain tasks
should be executed on a specific rank (or group or ranks).
To this end, Tarantella provides a number of functions to identify the rank ID
and allow users to add customized behavior based on rank, as decribed in this
:ref:`section<ranks-label>`.

In the case of the Transformer model, we want to use the rank information to
perform several tasks:

* print logging messages

.. code-block:: python

    if tnt.is_master_rank():
      logging.info("Start train")

* distribute datasets manually among participating devices
* execute other models, such as a modified, serial version of the Tarantella model for :ref:`inference<inference-master-rank-label>`
* enable certain callbacks only on one rank (e.g., profiling callbacks)

.. code-block:: python

    if tnt.is_master_rank():
      if self.flags_obj.enable_time_history:
        time_callback = keras_utils.TimeHistory(self.params["batch_size"],
                                                self.params["num_sentences"],
                                                logdir = None)
        callbacks.append(time_callback)

Such callbacks only collect local data corresponding to the specific rank where they are executed.
In this example, the `TimeHistory` callback will measure timings only on the `master_rank`. While
iteration and epoch runtimes should be the same on all ranks (as all ranks train in sync), other
metrics such as accuracy will only be computed based on the local data available to the rank.


.. _manually-distributed-datasets-label:

Using manually-distributed datasets
"""""""""""""""""""""""""""""""""""

Typically, it is the task of the framework to automatically handle batched
datasets, such that each rank only processes its share of the data, as explained in
the :ref:`Quick Start guide<using-distributed-datasets-label>`.

However, there are complex scenarios when the user might prefer to manually build the
dataset slices corresponding to each rank.
Tarantella allows the user to disable the automatic distribution mechanism
by passing ``tnt_distribute_dataset = False`` to the ``model.fit`` function.

This is how it is done in the case of the Transformer:

.. code-block:: python

    history = self.train_model.fit(train_ds,
                                   callbacks = callbacks,
                                   tnt_distribute_dataset = False,
                                   initial_epoch = epoch,
                                   epochs = epoch + min(self.params["epochs_between_evals"],
                                                       self.params["train_epochs"]-epoch),
                                   verbose = 2)

Also note the use of ``initial_epoch`` and ``epochs``. This combination of parameters
is necessary to allow evaluation rounds in between training epochs, when a validation
dataset cannot be simply passed to ``model.fit``.
In particular, our transformer implementation features a different model for
inference, as described :ref:`below<mixed-models-label>`.

Now that automatic distribution is disabled, let us take a look at how to split
the dataset manually among devices.
The input data processing is implemented in
`data_pipeline.py
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/data_pipeline.py>`_.

In the case of the Transformer model, the global ``batch_size`` stands for the total
number of input tokens processed in a single iteration.
However, as the training is performed in (fixed-sized) sentences, our global
``batch_size`` used for training will be in fact the number of sentences comprised
in such a batch.

Furthermore, we need to divide the number of sentences across ranks, such that
each rank can work on a separated shard of ``micro_batch_size`` sentences.
Finally, the dataset itself needs to be batched using the ``micro_batch_size`` and
each device instructed to select its own shard:

.. code-block:: python

  number_batch_sentences = batch_size // max_length

  micro_batch_size = number_batch_sentences // num_ranks

  # Batch the sentences and select only the shard (subset)
  # corresponding to the current rank
  dataset = dataset.padded_batch(micro_batch_size,
                                ([max_length], [max_length]),
                                drop_remainder=True)
  dataset = dataset.shard(num_ranks, rank)



.. _mixed-models-label:

Mixing Keras and Tarantella models
""""""""""""""""""""""""""""""""""

An essential aspect of the Transformer model is that it operates on slightly different
model versions during training and inference.
While in training the model works on encoded tokens, inference requires translation
to and from plain text. Thus, the model needs to use modified input and output layers
for each of these tasks.

To illustrate the way a Tarantella model can work alongside a typical Keras model, we
only execute the training phase on the Transformer within a (distributed) Tarantella
model.

Take a look at the
`model creation function
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/transformer/transformer_tnt.py#L53>`_.
It builds two different Keras models depending on whether training is enabled or not,
both of them based on the same `internal model` (i.e., using the same learned weights).

Now, when initializing our Transformer task, we only wrap one of the models as a ``tnt.Model``:

.. code-block:: python

  # Transformer model used both as Tarantella model (in training) and as a serial
  # model for inference
  internal_model = transformer.Transformer(self.params, name="transformer_v2")

  # The train model includes an additional logits layer and a customized loss
  self.train_model = create_model(internal_model, self.params, is_train = True)
  # Enable distributed training
  self.train_model = tnt.Model(self.train_model)

  # The inference model is wrapped as a different Keras model that does not use labels
  self.predict_model = create_model(internal_model, self.params, is_train = False)

Training can now proceed as usual, by only calling the ``fit`` method on our ``train_model``.
We can however design our training loop to stop every ``epochs_between_evals`` epochs,
evaluate the training accuracy using the serial ``predict_model``, and then continue
from where it left off.

.. _inference-master-rank-label:

.. code-block:: python

  for epoch in range(0, self.params["train_epochs"], self.params["epochs_between_evals"]):
    # as our dataset is distributed manually, disable the automatic Tarantella distribution
    history = self.train_model.fit(train_ds,
                                   callbacks = callbacks,
                                   tnt_distribute_dataset = False,
                                   initial_epoch = epoch,
                                   epochs = epoch + min(self.params["epochs_between_evals"],
                                                        self.params["train_epochs"]-epoch),
                                   verbose = 2)

    if tnt.is_master_rank():
      eval_stats = self.eval()

The ``self.eval()`` method performs the translation on the test dataset using the
standard Keras ``predict_model``.

.. code-block:: python

  def eval(self):
    ...
    uncased_score, cased_score = transformer_main.evaluate_and_log_bleu(
                                                  self.predict_model,
                                                  self.params,
                                                  self.flags_obj.bleu_source,
                                                  self.flags_obj.bleu_ref,
                                                  self.flags_obj.vocab_file)

A validation dataset can be provided in the form of a pair of input files specified
at the command line as  ``bleu_source`` and ``bleu_ref``.
If the validation dataset exists, the evaluation method will compute and log the
corresponding BLEU scores (both case-sensitive and case-insensitive) serially.