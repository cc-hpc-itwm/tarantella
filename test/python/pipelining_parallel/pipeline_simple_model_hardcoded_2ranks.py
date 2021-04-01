import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
from hardcoded_model import *

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
import pytest

number_tags = 2
num_micro_batches = 2

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
    p_0_shared_m_0_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_0_real_input_0")
    p_0_shared_m_1_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_1_real_input_0")
    p_0_shared_m_0_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_0_send_tag_0", dtype=tf.int32)
    p_0_shared_m_0_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_0_send_tag_1", dtype=tf.int32)
    p_0_shared_m_1_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_1_send_tag_0", dtype=tf.int32)
    p_0_shared_m_1_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_1_send_tag_1", dtype=tf.int32)
    p_0_shared_input_seq = keras.Input(shape=(1,), name = "p_0_seq_input")

    p_0_shared_core_inputs = [p_0_shared_m_0_input_0, p_0_shared_m_1_input_0]
    p_0_shared_recv_tags = []
    p_0_shared_send_tags = [p_0_shared_m_0_send_tag_0, p_0_shared_m_0_send_tag_1,
                            p_0_shared_m_1_send_tag_0, p_0_shared_m_1_send_tag_1]
    p_0_shared_start_seq = [p_0_shared_input_seq]

    p_0_m_0_outputs = p_0_shared_model(p_0_shared_core_inputs[0:1] + p_0_shared_send_tags[0:2] + p_0_shared_start_seq)
    p_0_m_1_outputs = p_0_shared_model(p_0_shared_core_inputs[1:2] + p_0_shared_send_tags[2:4] + [p_0_m_0_outputs[-1]])

    p_0_outputs = p_0_m_0_outputs[:-1] + p_0_m_1_outputs[:-1] + [p_0_m_1_outputs[-1]]
    p_0_outputs[0] = tnt_layers.IdentityLayer(name="p_0_m_0_edge_output_0")(p_0_outputs[0])
    p_0_outputs[1] = tnt_layers.IdentityLayer(name="p_0_m_0_edge_output_1")(p_0_outputs[1])
    p_0_outputs[2] = tnt_layers.IdentityLayer(name="p_0_m_1_edge_output_0")(p_0_outputs[2])
    p_0_outputs[3] = tnt_layers.IdentityLayer(name="p_0_m_1_edge_output_1")(p_0_outputs[3])
    p_0_outputs[4] = tnt_layers.IdentityLayer(name="p_0_seq_output")(p_0_outputs[4])
    p_0_inputs = p_0_shared_core_inputs + p_0_shared_recv_tags + p_0_shared_send_tags + p_0_shared_start_seq
    p_0 = keras.Model(inputs=p_0_inputs, outputs=p_0_outputs, name="p_0")
    return p_0

  elif rank == p_1_rank:
    p_1_shared_model = shared_model
    # --- microbatched model on partition 1
    p_1_shared_m_0_input_0 = keras.Input(shape=p_1_shared_model.inputs[0].shape[1:], name="p_1_m_0_edge_input_0")
    p_1_shared_m_0_input_1 = keras.Input(shape=p_1_shared_model.inputs[1].shape[1:], name="p_1_m_0_edge_input_1")
    p_1_shared_m_1_input_0 = keras.Input(shape=p_1_shared_model.inputs[0].shape[1:], name="p_1_m_1_edge_input_0")
    p_1_shared_m_1_input_1 = keras.Input(shape=p_1_shared_model.inputs[1].shape[1:], name="p_1_m_1_edge_input_1")
    p_1_shared_m_0_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_0_recv_tag_0", dtype=tf.int32)
    p_1_shared_m_0_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_0_recv_tag_1", dtype=tf.int32)
    p_1_shared_m_1_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_1_recv_tag_0", dtype=tf.int32)
    p_1_shared_m_1_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_1_recv_tag_1", dtype=tf.int32)
    p_1_shared_input_seq = keras.Input(shape=(1,), name = "p_1_seq_input")

    p_1_shared_core_inputs = [p_1_shared_m_0_input_0, p_1_shared_m_0_input_1,
                              p_1_shared_m_1_input_0, p_1_shared_m_1_input_1]
    p_1_shared_recv_tags = [p_1_shared_m_0_recv_tag_0, p_1_shared_m_0_recv_tag_1,
                            p_1_shared_m_1_recv_tag_0, p_1_shared_m_1_recv_tag_1]
    p_1_shared_send_tags = []
    p_1_shared_start_seq = [p_1_shared_input_seq]

    p_1_m_0_outputs = p_1_shared_model(p_1_shared_core_inputs[0:2] + p_1_shared_recv_tags[0:2] + p_1_shared_start_seq)
    p_1_m_1_outputs = p_1_shared_model(p_1_shared_core_inputs[2:4] + p_1_shared_recv_tags[2:4] + [p_1_m_0_outputs[-1]])

    p_1_outputs = p_1_m_0_outputs[:-1] + p_1_m_1_outputs[:-1] + [p_1_m_1_outputs[-1]]
    p_1_outputs[0] = tnt_layers.IdentityLayer(name="p_1_m_0_real_output_0")(p_1_outputs[0])
    p_1_outputs[1] = tnt_layers.IdentityLayer(name="p_1_m_1_real_output_0")(p_1_outputs[1])
    p_1_outputs[2] = tnt_layers.IdentityLayer(name="p_1_seq_output")(p_1_outputs[2])
    p_1_inputs = p_1_shared_core_inputs + p_1_shared_recv_tags + p_1_shared_send_tags + p_1_shared_start_seq
    p_1 = keras.Model(inputs=p_1_inputs, outputs=p_1_outputs, name="p_1")
    return p_1


@pytest.fixture(autouse=True)
def setup_tf_threading_before_tests():
  # at least as many parallel ops as connection IDs are needed to ensure the (blocking) send
  # operation on the last micro-batches can make progress
  if tf.config.threading.get_inter_op_parallelism_threads() < number_connections:
    tf.config.threading.set_inter_op_parallelism_threads(number_connections)
  yield

@pytest.mark.tfversion(['2.2', '2.3'])
class TestPipelineSimpleModel:

  @pytest.mark.parametrize("batch_size", [34])
  @pytest.mark.parametrize("num_batches", [10])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_train(self, batch_size, num_batches, number_epochs):
    assert tnt.get_size() == number_partitions
    micro_batch_size = batch_size // num_micro_batches

    ### CREATE MODEL
    pipeline_communicator = get_pipeline_communicator(micro_batch_size = micro_batch_size,
                                                      num_micro_batches = num_micro_batches)

    core_model = get_partitioned_core_model()
    shared_model = get_partitioned_shared_model(core_model, pipeline_communicator, micro_batch_size)
    microbatched_model = get_partitioned_model(shared_model)

    ### LOAD DATASETS
    ds = load_datasets(batch_size, num_batches, 0, num_micro_batches, core_model)

    ### MODEL COMPILE/TRAIN (on each rank individually)
    # single rank model
    sgd = keras.optimizers.SGD(learning_rate)
    if rank == master_rank:
      print("\nTraining reference model")
      reference_model = get_reference_model()
      reference_model.compile(optimizer = sgd,
                              loss = keras.losses.SparseCategoricalCrossentropy(),
                              metrics = [keras.metrics.SparseCategoricalAccuracy()])
      reference_history = reference_model.fit(ds["reference_train_dataset"],
                                              epochs = number_epochs,
                                              verbose = 0)

    # pipelined model
    if rank == p_0_rank:
      partition_losses = {"p_0_m_0_edge_output_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_0_edge_output_1" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_edge_output_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_edge_output_1" : tnt_losses.ZeroLoss(),
                          "p_0_seq_output" : tnt_losses.ZeroLoss()}
      partition_loss_weights = None
      partition_metrics = None
    if rank == p_1_rank:
      partition_losses = {"p_1_m_0_real_output_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_m_1_real_output_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_seq_output" : tnt_losses.ZeroLoss()}
      partition_loss_weights = {"p_1_m_0_real_output_0" : 1./num_micro_batches,
                                "p_1_m_1_real_output_0" : 1./num_micro_batches,
                                "p_1_seq_output" : 0.}
      partition_metrics = {"p_1_m_0_real_output_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_m_1_real_output_0" : keras.metrics.SparseCategoricalAccuracy()}

    microbatched_model.compile(optimizer = sgd,
                               loss = partition_losses,
                               loss_weights = partition_loss_weights,
                               metrics = partition_metrics)
    pipeline_history = microbatched_model.fit(ds["partition_train_dataset"],
                                              epochs = number_epochs,
                                              verbose = 0)
    if rank == master_rank:
      check_histories_match(reference_history, pipeline_history, num_micro_batches)


  @pytest.mark.parametrize("batch_size", [64])
  @pytest.mark.parametrize("num_batches", [200])
  @pytest.mark.parametrize("num_test_batches", [100])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_train_and_evaluate(self, batch_size, num_batches, num_test_batches, number_epochs):
    assert tnt.get_size() == number_partitions
    micro_batch_size = batch_size // num_micro_batches

    ### CREATE MODEL
    pipeline_communicator = get_pipeline_communicator(micro_batch_size = micro_batch_size,
                                                      num_micro_batches = num_micro_batches)

    core_model = get_partitioned_core_model()
    shared_model = get_partitioned_shared_model(core_model, pipeline_communicator, micro_batch_size)
    microbatched_model = get_partitioned_model(shared_model)

    ### LOAD DATASETS
    ds = load_datasets(batch_size, num_batches, num_test_batches, num_micro_batches, core_model)

    ### MODEL COMPILE/TRAIN (on each rank individually)
    # single rank model
    sgd = keras.optimizers.SGD(learning_rate)
    if rank == master_rank:
      print("\nTraining reference model")
      reference_model = get_reference_model()
      reference_model.compile(optimizer = sgd,
                              loss = keras.losses.SparseCategoricalCrossentropy(),
                              metrics = [keras.metrics.SparseCategoricalAccuracy()])
      reference_history = reference_model.fit(ds["reference_train_dataset"],
                                              validation_data = ds["reference_val_dataset"],
                                              epochs = number_epochs,
                                              verbose = 0)
      reference_result = reference_model.evaluate(ds["reference_test_dataset"],
                                                  verbose = 0)

    # pipelined model
    if rank == p_0_rank:
      partition_losses = {"p_0_m_0_edge_output_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_0_edge_output_1" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_edge_output_0" : tnt_losses.ZeroLoss(),
                          "p_0_m_1_edge_output_1" : tnt_losses.ZeroLoss(),
                          "p_0_seq_output" : tnt_losses.ZeroLoss()}
      partition_loss_weights = None
      partition_metrics = None
    if rank == p_1_rank:
      partition_losses = {"p_1_m_0_real_output_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_m_1_real_output_0" : keras.losses.SparseCategoricalCrossentropy(),
                          "p_1_seq_output" : tnt_losses.ZeroLoss()}
      partition_loss_weights = {"p_1_m_0_real_output_0" : 1./num_micro_batches,
                                "p_1_m_1_real_output_0" : 1./num_micro_batches,
                                "p_1_seq_output" : 0.}
      partition_metrics = {"p_1_m_0_real_output_0" : keras.metrics.SparseCategoricalAccuracy(),
                           "p_1_m_1_real_output_0" : keras.metrics.SparseCategoricalAccuracy()}

    microbatched_model.compile(optimizer = sgd,
                               loss = partition_losses,
                               loss_weights = partition_loss_weights,
                               metrics = partition_metrics)
    pipeline_history = microbatched_model.fit(ds["partition_train_dataset"],
                           validation_data = ds["partition_val_dataset"],
                           epochs = number_epochs,
                           verbose = 0)
    pipeline_result = microbatched_model.evaluate(ds["partition_test_dataset"],
                                                  verbose = 0)
    if rank == master_rank:
      check_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_validation_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_predictions_match(reference_result, pipeline_result, num_micro_batches)

