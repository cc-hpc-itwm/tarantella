from models import mnist_models as mnist
import utilities as util

import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched

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

def get_microbatched_dataset(samples, labels, num_micro_batches, micro_batch_size, core_model):
  if rank == p_0_rank:
    y_zero_loss = tf.data.Dataset.from_tensors(np.zeros(1)).repeat()

    partition_samples = [tf.data.Dataset.from_tensor_slices(samples)]
    partition_labels = [y_zero_loss, y_zero_loss]

    recv_connection_ids = [] # recv connection ids for each RecvLayer
    send_connection_ids = [0, 1] # send connection ids for each SendLayer
    partition_id = p_0_id
  elif rank == p_1_rank:
    # TODO:
    # Does the fake datasets/labels need a finite size, when used in a "middle" partition,
    # so `fit` would know, when to stop training?
    x_comm_0 = tf.data.Dataset.from_tensors(np.zeros(core_model.inputs[0].shape[1:])).repeat()
    x_comm_1 = tf.data.Dataset.from_tensors(np.zeros(core_model.inputs[1].shape[1:])).repeat()

    partition_samples = [x_comm_0, x_comm_1]
    partition_labels = [tf.data.Dataset.from_tensor_slices(labels)]

    recv_connection_ids = [0, 1]
    send_connection_ids = []
    partition_id = p_1_id

  return pipelining.create_micro_batched_dataset(samples = partition_samples,
                                                 labels = partition_labels,
                                                 recv_connection_ids = recv_connection_ids,
                                                 send_connection_ids = send_connection_ids,
                                                 partition_id = partition_id,
                                                 num_micro_batches = num_micro_batches,
                                                 micro_batch_size = micro_batch_size)


def check_histories_match(reference_history, pipeline_history, prefix = ""):
  loss_name = prefix + 'loss'
  metric_name = 'sparse_categorical_accuracy'

  for i in range(len(reference_history.history[loss_name])):
    assert np.allclose(reference_history.history[loss_name], pipeline_history.history[loss_name])
    assert np.allclose(reference_history.history[prefix + metric_name][i],
                       (pipeline_history.history[prefix + 'p_1_m_0_o_0_' + metric_name][i] +
                        pipeline_history.history[prefix + 'p_1_m_1_o_0_' + metric_name][i])/2)

def check_validation_histories_match(reference_history, pipeline_history):
  check_histories_match(reference_history, pipeline_history, prefix = "val_")

def check_predictions_match(reference_results, pipeline_results):
  assert np.allclose(reference_results[0], pipeline_results[0])
  assert np.allclose(reference_results[1], (pipeline_results[-2] + pipeline_results[-3])/2)

@pytest.fixture(autouse=True)
def setup_tf_threading_before_tests():
  if tf.config.threading.get_inter_op_parallelism_threads() < number_connections:
    tf.config.threading.set_inter_op_parallelism_threads(number_connections)
  yield

@pytest.mark.tfversion(['2.2', '2.3'])
class TestPipelineSimpleModel:

  @pytest.mark.parametrize("num_micro_batches", [2, 1, 3])
  @pytest.mark.parametrize("batch_size", [34])
  @pytest.mark.parametrize("num_batches", [10])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_train(self, num_micro_batches, batch_size, num_batches, number_epochs):
    # at least as many parallel ops as connection IDs are needed to ensure the (blocking) send
    # operation on the last micro-batches can make progress
    assert tf.config.threading.get_inter_op_parallelism_threads() >= number_connections
    assert tnt.get_size() == number_partitions
    train_size = num_batches * batch_size
    micro_batch_size = batch_size // num_micro_batches

    ### CREATE MODEL
    partition_table = { 0 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize),
                        1 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize) }
    ppl_comm = tnt.PipelineCommunicator(partition_table, num_micro_batches)


    core_model = get_partitioned_core_model()
    
    ### LOAD DATASETS
    (x_train, y_train), _, _ = mnist.load_mnist_dataset(train_size, 0, 0)
    train_dataset_reference = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                              .shuffle(len(x_train), shuffle_seed)

    partition_train_dataset = get_microbatched_dataset(x_train, y_train,
                                                       num_micro_batches, micro_batch_size,
                                                       core_model) \
                              .shuffle(len(x_train), shuffle_seed)

    if rank == p_0_rank:
      partition_info = pinfo.PartitionInfo(p_0_id, core_model)

      in_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
      partition_info.real_input_infos = [in_0]
      partition_info.edge_input_infos = []
      partition_info.real_output_infos = []

      out_edge_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
      out_edge_1 = pinfo.EndpointInfo(1, core_model.outputs[1].shape, tf.float32)
      partition_info.edge_output_infos = [out_edge_0, out_edge_1]

      losses = {}
      metrics = {}

    elif rank == p_1_rank:
      partition_info = pinfo.PartitionInfo(p_1_id, core_model)
      partition_info.real_input_infos = []

      in_edge_0 = pinfo.EndpointInfo(0, core_model.inputs[0].shape, tf.float32)
      in_edge_1 = pinfo.EndpointInfo(1, core_model.inputs[1].shape, tf.float32)
      partition_info.edge_input_infos = [in_edge_0, in_edge_1]

      out_0 = pinfo.EndpointInfo(0, core_model.outputs[0].shape, tf.float32)
      partition_info.real_output_infos = [out_0]
      partition_info.edge_output_infos = []

      losses = {0 : keras.losses.SparseCategoricalCrossentropy()}
      metrics = {0 : keras.metrics.SparseCategoricalAccuracy()}

    shared_model_builder = shared.SharedModelBuilder(partition_info, core_model, ppl_comm, micro_batch_size)
    shared_model = shared_model_builder.get_model()

    microbatched_model_builder = microbatched.MicrobatchedModelBuilder(partition_info, shared_model,
                                                              micro_batch_size, num_micro_batches)
    microbatched_model = microbatched_model_builder.get_model()

    microbatched_model.compile(optimizer = keras.optimizers.SGD(learning_rate),
                               loss = microbatched_model_builder.get_losses(losses),
                               #loss_weights = partition_loss_weights,
                               metrics = microbatched_model_builder.get_metrics(metrics))
    # keras.utils.plot_model(core_model, f"partition_{partition_info.partition_id}_core.png", show_shapes=True)
    # keras.utils.plot_model(shared_model, f"partition_{partition_info.partition_id}_shared.png", show_shapes=True)
    # keras.utils.plot_model(microbatched_model, f"partition_{partition_info.partition_id}_microbatched.png", show_shapes=True)
