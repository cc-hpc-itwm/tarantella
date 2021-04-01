import tarantella as tnt
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched
from hardcoded_model import *

import tensorflow as tf
import tensorflow.keras as keras
import pytest

import logging

@pytest.fixture(autouse=True)
def setup_tf_threading_before_tests():
  # at least as many parallel ops as connection IDs are needed to ensure the (blocking) send
  # operation on the last micro-batches can make progress
  if tf.config.threading.get_inter_op_parallelism_threads() < number_connections:
    tf.config.threading.set_inter_op_parallelism_threads(number_connections)
  yield

@pytest.mark.tfversion(['2.2', '2.3'])
class TestPipelineSimpleModel:

  @pytest.mark.parametrize("num_micro_batches", [2, 1, 3])
  @pytest.mark.parametrize("batch_size", [60, 36])
  @pytest.mark.parametrize("num_batches", [200])
  @pytest.mark.parametrize("num_test_batches", [100])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_train(self, num_micro_batches, batch_size, num_batches, num_test_batches, number_epochs):
    assert tnt.get_size() == number_partitions
    micro_batch_size = batch_size // num_micro_batches

    # create pipelined model and load datasets
    pipeline_communicator = get_pipeline_communicator(micro_batch_size = micro_batch_size,
                                                      num_micro_batches = num_micro_batches)
    core_model = get_partitioned_core_model()
    
    partition_info = get_partition_info(core_model)
    shared_model_builder = shared.SharedModelBuilder(partition_info, core_model, pipeline_communicator, micro_batch_size)
    shared_model = shared_model_builder.get_model()

    microbatched_model_builder = microbatched.MicrobatchedModelBuilder(partition_info, shared_model,
                                                                       micro_batch_size, num_micro_batches)
    microbatched_model = microbatched_model_builder.get_model()
    ds = load_datasets(batch_size, num_batches, num_test_batches, num_micro_batches, core_model)

    # reference model
    sgd = keras.optimizers.SGD(learning_rate)
    if rank == master_rank:
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
    # losses and metrics of the reference model
    reference_output_id = 0
    losses = {reference_output_id : keras.losses.SparseCategoricalCrossentropy()}
    metrics = {reference_output_id : keras.metrics.SparseCategoricalAccuracy()}

    microbatched_model.compile(optimizer = keras.optimizers.SGD(learning_rate),
                               loss = microbatched_model_builder.get_losses(losses),
                               loss_weights = microbatched_model_builder.get_loss_weights(),
                               metrics = microbatched_model_builder.get_metrics(metrics))

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