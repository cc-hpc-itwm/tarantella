from utilities import *
import tnt_keras_layers as tnt_layers

import os
import time
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress INFO messages from tf

import GPICommLib
import tarantella
tarantella.init()

class PlusOneLoss(tf.keras.losses.Loss):

  def __init__(self, name='test_loss'):
    super().__init__(name=name)

  def call(self, y_true, y_pred):
    @tf.custom_gradient
    def compute_loss(x):
      def grad(dy):
        return tf.ones_like(x) + x
      return tf.math.reduce_sum(x), grad

    fwd_bwd = tf.py_function(func=compute_loss,
                             inp=[y_pred],
                             Tout=tf.float32)
    return fwd_bwd

number_epochs = 5
num_micro_batches = 1
micro_batch_size = 2
tensor_len = 4
partition_table = {1: ((0,1), tensor_len * micro_batch_size * 4)} # size in bytes

ppl_comm = GPICommLib.PipelineCommunicator(tarantella.global_context, partition_table, num_micro_batches)

if tarantella.get_rank() == 0:
  # core model 0
  input_tags = keras.Input(shape=(2,), name="tags", dtype = tf.int32)

  input0 = keras.Input(shape=(tensor_len,), name = 'input')
  input_seq = keras.Input(shape = (1,), name='input_seq')
  output0 = tnt_layers.RemoveSeqInput()([input0, input_seq])
  output0 = tnt_layers.SendLayer(pipeline_communicator = ppl_comm)(output0[0], input_tags)
  output0 = tnt_layers.IdentityLayer(name = "result")(output0)
  model = keras.Model([input0, input_tags, input_seq], output0)
  
  tags_dataset = [[1, 0]]
  dataset = tf.data.Dataset.from_tensor_slices([[1,2,3,4], [11,22,33,44]]).batch(micro_batch_size)
  labels_dataset = tf.data.Dataset.from_tensor_slices([0, 0]).batch(micro_batch_size)
  loss = tnt_layers.ZeroLoss()

if tarantella.get_rank() == 1:
  # core model 1
  input_tags = keras.Input(shape=(2,), name="tags", dtype = tf.int32)

  input0 = keras.Input(shape=(tensor_len,), name = 'input')
  input_seq = keras.Input(shape = (1,), name='input_seq')
  output0 = tnt_layers.RemoveSeqInput()([input0, input_seq])
  output0 = tnt_layers.RecvLayer(pipeline_communicator = ppl_comm)(output0[0], input_tags)
  output0 = tnt_layers.IdentityLayer(name = "result")(output0)
  model = keras.Model([input0, input_tags, input_seq], output0)

  tags_dataset = [[1, 0]]
  dataset = tf.data.Dataset.from_tensor_slices([[9,9,9,9], [9,9,9,9]]).batch(micro_batch_size)
  labels_dataset = tf.data.Dataset.from_tensor_slices([3, 2]).batch(micro_batch_size)
  loss = PlusOneLoss()

def generator():
  input_seq_constant = 0
  for data, tag, label in zip(dataset, tags_dataset, labels_dataset):
    yield ({"input": data, "tags": tag, "input_seq": input_seq_constant}, 
           {"result": label})
final_dataset = tf.data.Dataset.from_generator(generator, output_types=({"input": tf.float32,
                                                                         "tags": tf.int32,
                                                                         "input_seq": tf.float32},
                                                                        {"result": tf.float32}))
final_dataset = final_dataset.repeat(number_epochs)

sgd = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer = sgd,
              loss = loss,
              experimental_run_tf_function = False)
model.fit(final_dataset,
          verbose = 1)
