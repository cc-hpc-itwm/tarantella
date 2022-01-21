import models.mnist_models as mnist
import utilities as util

import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.connection_info as cinfo
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
  util.set_tf_random_seed()
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
  util.set_tf_random_seed() # reset seed, so initial weights are same as for the reference model
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

def get_pipeline_communicator(num_micro_batches):
  connection_table = { 0 : cinfo.ConnectionInfo((p_0_rank, p_1_rank), fc_units * elem_type.itemsize),
                       1 : cinfo.ConnectionInfo((p_0_rank, p_1_rank), fc_units * elem_type.itemsize) }
  ppl_comm = tnt.PipelineCommunicator(connection_table, num_micro_batches)
  return ppl_comm

def get_partition_info(core_model):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo(p_0_id)

    in_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = {0 : in_0}
    partition_info.edge_input_infos = {}
    partition_info.real_output_infos = {}

    out_edge_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
    out_edge_1 = pinfo.EndpointInfo(1, core_model.outputs[1].shape, tf.float32)
    partition_info.edge_output_infos = {0 : out_edge_0, 1 : out_edge_1}

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo(p_1_id)
    partition_info.real_input_infos = {}

    in_edge_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
    in_edge_1 = pinfo.EndpointInfo(1, core_model.inputs[1].shape, tf.float32)
    partition_info.edge_input_infos = {0 : in_edge_0, 1 : in_edge_1}

    out_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = {0 : out_0}
    partition_info.edge_output_infos = {}
  return partition_info


def get_microbatched_dataset(samples, labels, micro_batch_size, num_micro_batches, partition_info):
  partition_samples = []
  partition_labels = []
  # assume all inputs are passed to the same start partition and
  # all outputs are generated on the same final partition
  if len(partition_info.get_real_ids(pinfo.EndpointDirection.inp)) > 0:
    partition_samples = [tf.data.Dataset.from_tensor_slices(samples)]
  if len(partition_info.get_real_ids(pinfo.EndpointDirection.out)) > 0:
    partition_labels = [tf.data.Dataset.from_tensor_slices(labels)]

  return pipelining.create_micro_batched_dataset(samples = partition_samples,
                                                 labels = partition_labels,
                                                 partition_info = partition_info,
                                                 num_micro_batches = num_micro_batches,
                                                 micro_batch_size = micro_batch_size,
                                                 dataset_size = len(samples))

def load_dataset_as_arrays(batch_size, num_batches, num_test_batches):
  train_size = num_batches * batch_size
  test_size = num_test_batches * batch_size
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist.load_mnist_dataset(
                                                          train_size, test_size, test_size)
  return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_microbatched_datasets(micro_batch_size, num_micro_batches, num_batches,
                               num_test_batches, partition_info):
  util.set_tf_random_seed()
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset_as_arrays(
                      micro_batch_size * num_micro_batches, num_batches, num_test_batches)
  train_dataset = get_microbatched_dataset(x_train, y_train, micro_batch_size,
                                           num_micro_batches, partition_info) \
                            .shuffle(len(x_train))
  val_dataset = get_microbatched_dataset(x_val, y_val, micro_batch_size,
                                         num_micro_batches, partition_info)
  test_dataset = get_microbatched_dataset(x_test, y_test, micro_batch_size,
                                          num_micro_batches, partition_info)
  return {"train" : train_dataset,
          "val"   : val_dataset,
          "test"  : test_dataset }

def load_reference_datasets(batch_size, num_batches, num_test_batches):
  util.set_tf_random_seed()
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset_as_arrays(
                                      batch_size, num_batches, num_test_batches)
  train_dataset = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                                                 .shuffle(len(x_train))
  val_dataset = util.create_dataset_from_arrays(x_val, y_val, batch_size=batch_size)
  test_dataset = util.create_dataset_from_arrays(x_test, y_test, batch_size=batch_size)
  return {"train" : train_dataset,
          "val"   : val_dataset,
          "test"  : test_dataset }

def check_histories_match(reference_history, pipeline_history, num_micro_batches, prefix = ""):
  loss_name = prefix + 'loss'
  metric_name = 'sparse_categorical_accuracy'
  output_id = 0
  partition_id = tnt.get_size() - 1 # compute metric only on the last partition

  for i in range(len(reference_history.history[loss_name])):
    # check loss matches
    assert np.allclose(reference_history.history[loss_name], pipeline_history.history[loss_name])

    # check metrics match 
    reference_metric_value = reference_history.history[prefix + metric_name][i]
    pipeline_metric_value = 0
    for m in range(num_micro_batches):
      pipeline_metric_value += \
        pipeline_history.history[f"{prefix}p_{partition_id}_m_{m}"
                                 f"_real_output_{output_id}_{metric_name}"][i]
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

def get_reference_compile_params():
  return {'optimizer' : keras.optimizers.SGD(0.01),
          'loss' : keras.losses.SparseCategoricalCrossentropy(),
          'metrics' : keras.metrics.SparseCategoricalAccuracy()}

def get_microbatched_compile_params(microbatched_model_builder):
  reference_output_id = 0
  reference_params = get_reference_compile_params()
  losses = {reference_output_id : reference_params['loss']}
  metrics = {reference_output_id : reference_params['metrics']}

  return {'optimizer' : reference_params['optimizer'],
          'loss' : microbatched_model_builder.get_losses(losses),
          'loss_weights' : microbatched_model_builder.get_loss_weights(),
          'metrics' : microbatched_model_builder.get_metrics(metrics)}

