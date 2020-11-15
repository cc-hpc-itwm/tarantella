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
schemes that allow for a seemless transition from basic serial training to distributed 
data parallelism using Tarantella.


Prerequisites
-------------

The tutorial models can be downloaded from the 
`Tnt Models repository <https://github.com/cc-hpc-itwm/tarantella_models>`_

.. code-block:: bash

    export TNT_MODELS_PATH=/your/installation/path
    cd ${TNT_MODELS_PATH}
    git clone https://github.com/cc-hpc-itwm/tarantella_models

To use these models, install the the following dependencies:

* TensorFlow 2.2.1
* Tarantella 0.6.0

For a step-by-step installation, follow the :ref:`installation-label` guide.
In the following we will assume that TensorFlow was installed in a ``conda`` 
environment called ``tarantella``.

Now we can install the final dependency,
`TensorFlow official Model Garden <https://github.com/tensorflow/models>`__:

.. code-block:: bash

    conda activate tarantella
    pip install tf-models-official==2.2.1


.. _resnet50-label:

ResNet-50
---------

Deep Residual Networks (ResNets) represented a breakthrough in the field of
computer vision, enabling deeper and more complex deep convolutional networks.
Introduced in [He]_, ResNet50 has become a standard model for image classification 
tasks, and has be shown to scale to very large number of nodes in data parallel 
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
``validation`` data performed in between training epochs.

Note the ``--batch_size`` parameter, which specifies the global batch size used in training.

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^
We will now look closer into the implementation of the ResNet-50 training scheme. 
The main training steps reside in the ``models/resnet/resnet50_tnt.py`` file.

The most important step in enabling data parallelism with Tarantella is 
to wrap the Keras model:

.. code-block:: python

    model = resnet_model.resnet50(num_classes=tf_imagenet_preprocessing.NUM_CLASSES)
    model = tnt.Model(model)

The following operations can be used for training the model serially, as they do not 
require any change.
In particular, the ImageNet dataset is loaded and preprocessed as follows:

.. code-block:: python

    train_dataset = imagenet_preprocessing.input_fn(is_training=True,
                                                    data_dir=flags_obj.data_dir,
                                                    batch_size=flags_obj.batch_size,
                                                    shuffle_seed = 42,
                                                    drop_remainder=True)

The 
`imagenet_preprocessing.input_fn
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/resnet/imagenet_preprocessing.py>`_
function takes the input files in ``data_dir``, loads the training samples and processes 
them into TensorFlow datasets.

The user only needs to pass the global ``batch_size`` value, and the Tarantella 
framework will ensure that the dataset is properly distributed among devices,
such that:

  * each device will process an independent set of samples
  * each device will group the samples into micro batches, where the micro-batch
    size will be computed as ``batch_size / num_devices``
  * each device will apply the same set of transformation to its imput samples as 
    specified in the ``input_fn`` function.

Before starting the training, the model is compiled to use a standard Keras optimizer 
and loss.

.. code-block:: python

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=(['sparse_categorical_accuracy']))

We provide flags to enable the most commonly used Keras ``callbacks``, such as 
the ``TensorBoard`` profiler, which can simply be passed to the ``fit`` function 
of the Tarantella model.

.. code-block:: python

    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir,
                                                    profile_batch=2))

If model checkpointing is required, it can be enabled through the ``ModelCheckpoint`` 
callback as usual (cf. :ref:`checkpointing models with Tarantella <checkpointing-via-callbacks-label>`).

.. code-block:: python

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))


There is no need for any further changes to proceed with training:

.. code-block:: python

    history = model.fit(train_dataset,
                        epochs=flags_obj.train_epochs,
                        callbacks=callbacks,
                        validation_data=validation_dataset,
                        validation_freq=flags_obj.epochs_between_evals,
                        verbose=1)

.. todo::

   Advanced topics:

   * scaling batch size with number of ranks (-> only mention here & link to advanced topics)
   * introduce learning rate warm up
   * introduce learning rate scaling (with #ranks)


.. _transformer-label:

Transformers
------------

The Transformer is a Deep Neural Network widely used in the field of natural language processing (NLP),
in particular for tasks such as machine translation.
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
                         --vocab_file=${WMT14_PATH}/vocab.ende.32768     
                         --bleu_ref=${WMT14_PATH}/newstest2014.de 
                         --bleu_source=${WMT14_PATH}/newstest2014.en 
                         --param_set=big 
                         --train_epochs=30
                         --batch_size=32736

The above command will select the ``big`` model implementation and train it
distributedly on the 8 specified devices.
To reach the target accuracy, [Vaswani]_ specifies that the model needs to be 
trained for ``30`` epochs.

The Transformer requires access to a vocabulary file, which contains all the
tokens derived from the dataset. This is provided as the ``vocab_file`` parameter
and is part of the pre-processed dataset.

After training, one round of evaluation is conducted using the ``newstest2014``
dataset to translate English sentences into German.

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^

The Transformer model itself is implemented and imported from the 
`TensorFlow Model Garden 
<https://github.com/tensorflow/models/tree/master/official/nlp/transformer>`__.
The training procedure and dataset loading and pre-processing do not require
extensive changes to work with Tarantella. However, we provide a simplified 
version to highlight the usage of Tarantella with Keras training loops.

Thus, the Keras transformer model is created in
``models/transformer/transformer_tnt.py`` and wrapped into a Tarantella model:

.. code-block:: python

    model = resnet_model.resnet50(num_classes=tf_imagenet_preprocessing.NUM_CLASSES)
    model = tnt.Model(model)

Data is loaded as follows, without any specific modification to trigger 
distributed training:

.. code-block:: python

    train_ds = data_pipeline.train_input_fn(self.params)

Here, the ``data_pipeline.train_input_fn`` reads in the dataset and applies a series 
of transformations to convert it into a batched set of sentences.
The advantage of using the *automatic dataset distribution* mechanism of Tarantella
is that users can reason about their I/O pipeline without taking care of the details
about how to distribute it.
Note however, that the batch size has to be a multiple of the number of ranks, so
that it can be efficiently divided into micro-batches.

Next, the user can also create callbacks, which can then be simply passed on to
the training function.

.. code-block:: python

  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.flags_obj.model_dir))

Finally, we can call ``model.fit`` to start distributed training on all devices:

.. code-block:: python

    history = model.fit(train_ds,
                        epochs=self.params["train_epochs"],
                        callbacks=callbacks,
                        verbose=1)

.. todo::

   Important points
   
   * Mixing Keras and Tarantella models

