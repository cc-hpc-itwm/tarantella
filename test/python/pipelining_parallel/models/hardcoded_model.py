import models.mnist_models as mnist
import utilities as util

import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
import pytest

### MODEL CONFIGURATION (on _all_ ranks)
fc_units = 100
num_mnist_classes = 10
shuffle_seed = 17
learning_rate = 0.01
elem_type = np.dtype(np.float32)

number_connections = 2
number_partitions = 2
p_0_id = 0
p_1_id = 1
# Results correctness is checked on the `master_rank`, which has to be on rank=0 to be able to
# forward the test exit code to `gaspi_run`
p_0_rank = 1
p_1_rank = 0
master_rank = p_1_rank
rank = tnt.get_rank()


def get_reference_model():
  tf.random.set_seed(42)
  reference_input = keras.Input(shape=(28,28,1,), name='reference_input')
  reference_x = layers.Flatten()(reference_input)
  reference_x = layers.Dense(fc_units, activation='relu', name='dense_relu')(reference_x)
  reference_output = layers.Dense(num_mnist_classes,
                                  activation='softmax',
                                  name='dense_softmax')(reference_x + reference_x)
  reference_model = keras.Model(inputs=reference_input, outputs=reference_output, name="reference_model")
  return reference_model

def get_partitioned_core_model():
  # --- core model on partition 0
  tf.random.set_seed(42) # reset seed, so initial weights are same as for the reference model
  p_0_core_input = keras.Input(shape=(28,28,1,)) # may be more than one
  p_0_core_x = layers.Flatten()(p_0_core_input)
  p_0_core_output_0 = layers.Dense(fc_units, activation='relu', name='dense_relu'+'_0')(p_0_core_x)
  p_0_core_output_1 = tnt_layers.IdentityLayer(name='dense_relu'+'_1')(p_0_core_output_0)
  p_0_core = keras.Model(inputs=p_0_core_input,
                        outputs=[p_0_core_output_0, p_0_core_output_1],
                        name="core_layers_p_0")

  # --- core model on partition 1
  p_1_core_input_0_shape = p_0_core.outputs[0].shape[1:]
  p_1_core_input_1_shape = p_0_core.outputs[1].shape[1:]
  p_1_core_input_0 = keras.Input(shape=p_1_core_input_0_shape) # TODO: Maybe add dtypes?
  p_1_core_input_1 = keras.Input(shape=p_1_core_input_1_shape)
  p_1_core_x = p_1_core_input_0 + p_1_core_input_1
  p_1_core_output_0 = layers.Dense(num_mnist_classes, activation='softmax', name='dense_softmax')(p_1_core_x)
  p_1_core = keras.Model(inputs=[p_1_core_input_0, p_1_core_input_1],
                        outputs=[p_1_core_output_0],
                        name="core_layers_p_1")
  if rank == p_0_rank:
    return p_0_core
  elif rank == p_1_rank:
    return p_1_core

def get_pipeline_communicator(micro_batch_size, num_micro_batches):
  partition_table = { 0 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize),
                      1 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize) }
  ppl_comm = tnt.PipelineCommunicator(partition_table, num_micro_batches)
  return ppl_comm

def get_partition_info(core_model):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo(p_0_id)

    in_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]
    partition_info.edge_input_infos = []
    partition_info.real_output_infos = []

    out_edge_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
    out_edge_1 = pinfo.EndpointInfo(1, core_model.outputs[1].shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0, out_edge_1]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo(p_1_id)
    partition_info.real_input_infos = []

    in_edge_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
    in_edge_1 = pinfo.EndpointInfo(1, core_model.inputs[1].shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0, in_edge_1]

    out_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = [out_0]
    partition_info.edge_output_infos = []
  return partition_info


def get_microbatched_dataset(samples, labels, micro_batch_size, num_micro_batches, core_model):
  if rank == p_0_rank:
    partition_samples = [tf.data.Dataset.from_tensor_slices(samples)]
    partition_labels = []
  elif rank == p_1_rank:
    partition_samples = []
    partition_labels = [tf.data.Dataset.from_tensor_slices(labels)]

  partition_info = get_partition_info(core_model)
  return pipelining.create_micro_batched_dataset(samples = partition_samples,
                                                 labels = partition_labels,
                                                 partition_info = partition_info,
                                                 num_micro_batches = num_micro_batches,
                                                 micro_batch_size = micro_batch_size)

def load_datasets(batch_size, num_batches, num_test_batches, num_micro_batches, core_model):
  train_size = num_batches * batch_size
  test_size = num_test_batches * batch_size
  micro_batch_size = batch_size // num_micro_batches

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist.load_mnist_dataset(
                                                                train_size, test_size, test_size)
  reference_train_dataset = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                            .shuffle(len(x_train), shuffle_seed)
  reference_val_dataset = util.create_dataset_from_arrays(x_val, y_val, batch_size=batch_size)
  reference_test_dataset = util.create_dataset_from_arrays(x_test, y_test, batch_size=batch_size)

  partition_train_dataset = get_microbatched_dataset(x_train, y_train,
                                                     micro_batch_size, num_micro_batches,
                                                     core_model) \
                            .shuffle(len(x_train), shuffle_seed)
  partition_val_dataset = get_microbatched_dataset(x_val, y_val,
                                                   micro_batch_size, num_micro_batches, core_model)
  partition_test_dataset = get_microbatched_dataset(x_test, y_test,
                                                    micro_batch_size, num_micro_batches, core_model)

  return { "reference_train_dataset" : reference_train_dataset,
           "reference_val_dataset"   : reference_val_dataset,
           "reference_test_dataset"  : reference_test_dataset,
           "partition_train_dataset" : partition_train_dataset,
           "partition_val_dataset"   : partition_val_dataset,
           "partition_test_dataset"  : partition_test_dataset }

def check_histories_match(reference_history, pipeline_history, num_micro_batches, prefix = ""):
  loss_name = prefix + 'loss'
  metric_name = 'sparse_categorical_accuracy'

  for i in range(len(reference_history.history[loss_name])):
    # check loss matches
    assert np.allclose(reference_history.history[loss_name], pipeline_history.history[loss_name])

    # check metrics match 
    reference_metric_value = reference_history.history[prefix + metric_name][i]
    pipeline_metric_value = 0
    for m in range(num_micro_batches):
      pipeline_metric_value += \
        pipeline_history.history[prefix + 'p_1_m_' + str(m) + '_real_output_0_' + metric_name][i]
    pipeline_metric_value = pipeline_metric_value / num_micro_batches
    assert np.allclose(reference_metric_value, pipeline_metric_value)

def check_validation_histories_match(reference_history, pipeline_history, num_micro_batches):
  check_histories_match(reference_history, pipeline_history, num_micro_batches, prefix = "val_")

def check_predictions_match(reference_results, pipeline_results, num_micro_batches):
  reference_loss = reference_results[0]
  pipeline_loss = pipeline_results[0]
  assert np.allclose(reference_loss, pipeline_loss)

  reference_accuracy = reference_results[1]
  pipeline_accuracy = np.sum(pipeline_results[-num_micro_batches:]) / num_micro_batches
  assert np.allclose(reference_accuracy, pipeline_accuracy)
