from models import mnist_models as mnist
import utilities as util

import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics
import tarantella.strategy.pipelining.utilities as pipelining

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

num_micro_batches = 2
number_tags = 2

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

def get_partitioned_shared_model(core_model, ppl_comm, micro_batch_size):
  if rank == p_0_rank:
    p_0_core = core_model
    # --- shared model on partition 0
    p_0_shared_input_0 = keras.Input(shape=(28,28,1,))
    p_0_shared_send_tag_0 = keras.Input(shape=(number_tags,), dtype=tf.int32)
    p_0_shared_send_tag_1 = keras.Input(shape=(number_tags,), dtype=tf.int32)
    p_0_shared_input_seq = keras.Input(shape=(1,)) # use to model sequential dependencies # TODO: Add dtype?!

    p_0_shared_core_inputs = [p_0_shared_input_0]
    p_0_shared_recv_tags = []
    p_0_shared_send_tags = [p_0_shared_send_tag_0, p_0_shared_send_tag_1]
    p_0_shared_start_seq = [p_0_shared_input_seq]

    p_0_shared_x = tnt_layers.RemoveSeqInput()(p_0_shared_core_inputs + p_0_shared_start_seq)
    p_0_shared_x = p_0_core(p_0_shared_x)

    p_0_shared_output_0 = tnt_layers.SendLayer(pipeline_communicator = ppl_comm)(
                                                    p_0_shared_x[0], p_0_shared_send_tags[0])
    p_0_shared_output_1 = tnt_layers.SendLayer(pipeline_communicator = ppl_comm)(
                                                    p_0_shared_x[1], p_0_shared_send_tags[1])
    p_0_shared_outputs = [p_0_shared_output_0, p_0_shared_output_1]
    p_0_shared_outputs = tnt_layers.AddSeqOutput(micro_batch_size = micro_batch_size)(p_0_shared_outputs)
    p_0_shared_inputs = p_0_shared_core_inputs + p_0_shared_recv_tags + p_0_shared_send_tags + p_0_shared_start_seq
    p_0_shared_model = keras.Model(inputs=p_0_shared_inputs, outputs=p_0_shared_outputs, name="p_0_shared")
    return p_0_shared_model

  elif rank == p_1_rank:
    p_1_core = core_model
    # --- shared model on partition 1
    p_1_shared_input_0 = keras.Input(shape = p_1_core.inputs[0].shape[1:])
    p_1_shared_input_1 = keras.Input(shape = p_1_core.inputs[1].shape[1:])
    p_1_shared_recv_tag_0 = keras.Input(shape=(number_tags,), dtype=tf.int32)
    p_1_shared_recv_tag_1 = keras.Input(shape=(number_tags,), dtype=tf.int32)
    p_1_shared_input_seq = keras.Input(shape=(1,))

    p_1_shared_core_inputs = [p_1_shared_input_0, p_1_shared_input_1]
    p_1_shared_recv_tags = [p_1_shared_recv_tag_0, p_1_shared_recv_tag_1]
    p_1_shared_send_tags = []
    p_1_shared_start_seq = [p_1_shared_input_seq]

    p_1_shared_x = tnt_layers.RemoveSeqInput()(p_1_shared_core_inputs + p_1_shared_start_seq)
    p_1_shared_recved_0 = tnt_layers.RecvLayer(pipeline_communicator = ppl_comm)(
                                               p_1_shared_x[0], p_1_shared_recv_tags[0])
    p_1_shared_recved_1 = tnt_layers.RecvLayer(pipeline_communicator = ppl_comm)(
                                               p_1_shared_x[1], p_1_shared_recv_tags[1])
    p_1_shared_outputs = p_1_core([p_1_shared_recved_0, p_1_shared_recved_1])

    p_1_shared_outputs = tnt_layers.AddSeqOutput(micro_batch_size=micro_batch_size)(p_1_shared_outputs)
    p_1_shared_inputs = p_1_shared_core_inputs + p_1_shared_recv_tags + p_1_shared_send_tags + p_1_shared_start_seq
    p_1_shared_model = keras.Model(inputs=p_1_shared_inputs, outputs=p_1_shared_outputs, name="p_1_shared")
    return p_1_shared_model

def get_partitioned_model(shared_model):
  if rank == p_0_rank:
    p_0_shared_model = shared_model
    # --- microbatched model on partition 0
    p_0_shared_m_0_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_0_i_0")
    p_0_shared_m_1_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_1_i_0")
    p_0_shared_m_0_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_0_s_0", dtype=tf.int32)
    p_0_shared_m_0_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_0_s_1", dtype=tf.int32)
    p_0_shared_m_1_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_1_s_0", dtype=tf.int32)
    p_0_shared_m_1_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_1_s_1", dtype=tf.int32)
    p_0_shared_input_seq = keras.Input(shape=(1,), name = "p_0_start_seq")

    p_0_shared_core_inputs = [p_0_shared_m_0_input_0, p_0_shared_m_1_input_0]
    p_0_shared_recv_tags = []
    p_0_shared_send_tags = [p_0_shared_m_0_send_tag_0, p_0_shared_m_0_send_tag_1,
                            p_0_shared_m_1_send_tag_0, p_0_shared_m_1_send_tag_1]
    p_0_shared_start_seq = [p_0_shared_input_seq]

    p_0_m_0_outputs = p_0_shared_model(p_0_shared_core_inputs[0:1] + p_0_shared_send_tags[0:2] + p_0_shared_start_seq)
    p_0_m_1_outputs = p_0_shared_model(p_0_shared_core_inputs[1:2] + p_0_shared_send_tags[2:4] + [p_0_m_0_outputs[-1]])

    p_0_outputs = p_0_m_0_outputs[:-1] + p_0_m_1_outputs[:-1] + [p_0_m_1_outputs[-1]]
    p_0_outputs[0] = tnt_layers.IdentityLayer(name="p_0_m_0_o_0")(p_0_outputs[0])
    p_0_outputs[1] = tnt_layers.IdentityLayer(name="p_0_m_0_o_1")(p_0_outputs[1])
    p_0_outputs[2] = tnt_layers.IdentityLayer(name="p_0_m_1_o_0")(p_0_outputs[2])
    p_0_outputs[3] = tnt_layers.IdentityLayer(name="p_0_m_1_o_1")(p_0_outputs[3])
    p_0_outputs[4] = tnt_layers.IdentityLayer(name="p_0_end_seq")(p_0_outputs[4])
    p_0_inputs = p_0_shared_core_inputs + p_0_shared_recv_tags + p_0_shared_send_tags + p_0_shared_start_seq
    p_0 = keras.Model(inputs=p_0_inputs, outputs=p_0_outputs, name="p_0")
    return p_0

  elif rank == p_1_rank:
    p_1_shared_model = shared_model
    # --- microbatched model on partition 1
    p_1_shared_m_0_input_0 = keras.Input(shape=p_1_shared_model.inputs[0].shape[1:], name="p_1_m_0_i_0")
    p_1_shared_m_0_input_1 = keras.Input(shape=p_1_shared_model.inputs[1].shape[1:], name="p_1_m_0_i_1")
    p_1_shared_m_1_input_0 = keras.Input(shape=p_1_shared_model.inputs[0].shape[1:], name="p_1_m_1_i_0")
    p_1_shared_m_1_input_1 = keras.Input(shape=p_1_shared_model.inputs[1].shape[1:], name="p_1_m_1_i_1")
    p_1_shared_m_0_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_0_r_0", dtype=tf.int32)
    p_1_shared_m_0_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_0_r_1", dtype=tf.int32)
    p_1_shared_m_1_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_1_r_0", dtype=tf.int32)
    p_1_shared_m_1_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_1_r_1", dtype=tf.int32)
    p_1_shared_input_seq = keras.Input(shape=(1,), name = "p_1_start_seq")

    p_1_shared_core_inputs = [p_1_shared_m_0_input_0, p_1_shared_m_0_input_1,
                              p_1_shared_m_1_input_0, p_1_shared_m_1_input_1]
    p_1_shared_recv_tags = [p_1_shared_m_0_recv_tag_0, p_1_shared_m_0_recv_tag_1,
                            p_1_shared_m_1_recv_tag_0, p_1_shared_m_1_recv_tag_1]
    p_1_shared_send_tags = []
    p_1_shared_start_seq = [p_1_shared_input_seq]

    p_1_m_0_outputs = p_1_shared_model(p_1_shared_core_inputs[0:2] + p_1_shared_recv_tags[0:2] + p_1_shared_start_seq)
    p_1_m_1_outputs = p_1_shared_model(p_1_shared_core_inputs[2:4] + p_1_shared_recv_tags[2:4] + [p_1_m_0_outputs[-1]])

    p_1_outputs = p_1_m_0_outputs[:-1] + p_1_m_1_outputs[:-1] + [p_1_m_1_outputs[-1]]
    p_1_outputs[0] = tnt_layers.IdentityLayer(name="p_1_m_0_o_0")(p_1_outputs[0])
    p_1_outputs[1] = tnt_layers.IdentityLayer(name="p_1_m_1_o_0")(p_1_outputs[1])
    p_1_outputs[2] = tnt_layers.IdentityLayer(name="p_1_end_seq")(p_1_outputs[2])
    p_1_inputs = p_1_shared_core_inputs + p_1_shared_recv_tags + p_1_shared_send_tags + p_1_shared_start_seq
    p_1 = keras.Model(inputs=p_1_inputs, outputs=p_1_outputs, name="p_1")
    return p_1

def get_microbatched_dataset(samples, labels, micro_batch_size, core_model):
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

  @pytest.mark.parametrize("batch_size", [34])
  @pytest.mark.parametrize("num_batches", [10])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_train(self, batch_size, num_batches, number_epochs):
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
    shared_model = get_partitioned_shared_model(core_model, ppl_comm, micro_batch_size)
    microbatched_model = get_partitioned_model(shared_model)

    ### LOAD DATASETS
    (x_train, y_train), _, _ = mnist.load_mnist_dataset(train_size, 0, 0)
    train_dataset_reference = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                              .shuffle(len(x_train), shuffle_seed)

    partition_train_dataset = get_microbatched_dataset(x_train, y_train,
                                                       micro_batch_size, core_model) \
                              .shuffle(len(x_train), shuffle_seed)

    ### MODEL COMPILE/TRAIN (on each rank individually)
    # single rank model
    sgd = keras.optimizers.SGD(learning_rate)
    if rank == master_rank:
      print("\nTraining reference model")
      reference_model = get_reference_model()
      reference_model.compile(optimizer = sgd,
                              loss = keras.losses.SparseCategoricalCrossentropy(),
                              metrics = [keras.metrics.SparseCategoricalAccuracy()])
      reference_history = reference_model.fit(train_dataset_reference,
                                              epochs = number_epochs,
                                              verbose = 0)

    # pipelined model
    if rank == p_0_rank:
      partition_losses = {"p_0_m_0_o_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_0_o_1" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_o_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_o_1" : tnt_losses.ZeroLoss(),
                          "p_0_end_seq" : tnt_losses.ZeroLoss()}
      partition_loss_weights = None
      partition_metrics = None
    if rank == p_1_rank:
      partition_losses = {"p_1_m_0_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_m_1_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_end_seq" : tnt_losses.ZeroLoss()}
      partition_loss_weights = {"p_1_m_0_o_0" : 1./num_micro_batches,
                                "p_1_m_1_o_0" : 1./num_micro_batches,
                                "p_1_end_seq" : 0.}
      partition_metrics = {"p_1_m_0_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_m_1_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_end_seq" : tnt_metrics.ZeroMetric()}

    microbatched_model.compile(optimizer = sgd,
                               loss = partition_losses,
                               loss_weights = partition_loss_weights,
                               metrics = partition_metrics)
    pipeline_history = microbatched_model.fit(partition_train_dataset,
                                              epochs = number_epochs,
                                              verbose = 0)
    if rank == master_rank:
      check_histories_match(reference_history, pipeline_history)


  @pytest.mark.parametrize("batch_size", [64])
  @pytest.mark.parametrize("num_batches", [200])
  @pytest.mark.parametrize("num_test_batches", [100])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_train_and_evaluate(self, batch_size, num_batches, num_test_batches, number_epochs):
    assert tnt.get_size() == number_partitions
    train_size = num_batches * batch_size
    test_size = num_test_batches * batch_size
    micro_batch_size = batch_size // num_micro_batches

    ### CREATE MODEL
    partition_table = { 0 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize),
                        1 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * elem_type.itemsize) }
    ppl_comm = tnt.PipelineCommunicator(partition_table, num_micro_batches)

    core_model = get_partitioned_core_model()
    shared_model = get_partitioned_shared_model(core_model, ppl_comm, micro_batch_size)
    microbatched_model = get_partitioned_model(shared_model)

    ### LOAD DATASETS
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist.load_mnist_dataset(
                                                                  train_size, test_size, test_size)
    train_dataset_reference = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                              .shuffle(len(x_train), shuffle_seed)
    val_dataset_reference = util.create_dataset_from_arrays(x_val, y_val, batch_size=batch_size)
    test_dataset_reference = util.create_dataset_from_arrays(x_test, y_test, batch_size=batch_size)

    partition_train_dataset = get_microbatched_dataset(x_train, y_train,
                                                       micro_batch_size, core_model) \
                              .shuffle(len(x_train), shuffle_seed)
    partition_val_dataset = get_microbatched_dataset(x_val, y_val,
                                                     micro_batch_size, core_model)
    partition_test_dataset = get_microbatched_dataset(x_test, y_test,
                                                      micro_batch_size, core_model)

    ### MODEL COMPILE/TRAIN (on each rank individually)
    # single rank model
    sgd = keras.optimizers.SGD(learning_rate)
    if rank == master_rank:
      print("\nTraining reference model")
      reference_model = get_reference_model()
      reference_model.compile(optimizer = sgd,
                              loss = keras.losses.SparseCategoricalCrossentropy(),
                              metrics = [keras.metrics.SparseCategoricalAccuracy()])
      reference_history = reference_model.fit(train_dataset_reference,
                                              validation_data = val_dataset_reference,
                                              epochs = number_epochs,
                                              verbose = 0)
      reference_result = reference_model.evaluate(test_dataset_reference,
                                                  verbose = 0)

    # pipelined model
    if rank == p_0_rank:
      partition_losses = {"p_0_m_0_o_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_0_o_1" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_o_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_o_1" : tnt_losses.ZeroLoss(),
                          "p_0_end_seq" : tnt_losses.ZeroLoss()}
      partition_loss_weights = None
      partition_metrics = None
    if rank == p_1_rank:
      partition_losses = {"p_1_m_0_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_m_1_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_end_seq" : tnt_losses.ZeroLoss()}
      partition_loss_weights = {"p_1_m_0_o_0" : 1./num_micro_batches,
                                "p_1_m_1_o_0" : 1./num_micro_batches,
                                "p_1_end_seq" : 0.}
      partition_metrics = {"p_1_m_0_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_m_1_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_end_seq" : tnt_metrics.ZeroMetric()}

    microbatched_model.compile(optimizer = sgd,
                               loss = partition_losses,
                               loss_weights = partition_loss_weights,
                               metrics = partition_metrics)
    pipeline_history = microbatched_model.fit(partition_train_dataset,
                           validation_data = partition_val_dataset,
                           epochs = number_epochs,
                           verbose = 0)
    pipeline_result = microbatched_model.evaluate(partition_test_dataset,
                                                  verbose = 0)
    if rank == master_rank:
      check_histories_match(reference_history, pipeline_history)
      check_validation_histories_match(reference_history, pipeline_history)
      check_predictions_match(reference_result, pipeline_result)

