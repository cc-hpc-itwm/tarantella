import tarantella as tnt
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched
from models.hardcoded_model import *

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

@pytest.mark.min_tfversion('2.2')
class TestPipelineSimpleModel:

  @pytest.mark.parametrize("num_micro_batches", [2, 1, 3])
  @pytest.mark.parametrize("batch_size", [60, 36])
  @pytest.mark.parametrize("num_batches", [200])
  @pytest.mark.parametrize("num_test_batches", [100])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_train(self, num_micro_batches, batch_size, num_batches, num_test_batches, number_epochs):
    assert tnt.get_size() == number_partitions
    fit_params = {'epochs' : number_epochs, 'shuffle' : False, 'verbose' : 0}
    micro_batch_size = batch_size // num_micro_batches

    # create pipelined model and load datasets
    pipeline_communicator = get_pipeline_communicator(num_micro_batches)
    pipeline_communicator.setup_infrastructure(micro_batch_size)
    core_model = get_partitioned_core_model()
    
    partition_info = get_partition_info(core_model)
    shared_model_builder = shared.SharedModelBuilder(partition_info, core_model, pipeline_communicator, micro_batch_size)
    shared_model = shared_model_builder.get_model()

    microbatched_model_builder = microbatched.MicrobatchedModelBuilder(partition_info, shared_model,
                                                                       micro_batch_size, num_micro_batches)
    microbatched_model = microbatched_model_builder.get_model()
    microbatched_ds = load_microbatched_datasets(micro_batch_size, num_micro_batches,
                                                 num_batches, num_test_batches, partition_info)

    # reference model
    if rank == master_rank:
      reference_model = get_reference_model()
      reference_model.compile(**get_reference_compile_params())

      reference_ds = load_reference_datasets(batch_size, num_batches, num_test_batches)
      reference_history = reference_model.fit(reference_ds["train"],
                                              validation_data = reference_ds["val"],
                                              **fit_params)
      reference_result = reference_model.evaluate(reference_ds["test"], verbose = 0)

    # pipelined model
    microbatched_model.compile(**get_microbatched_compile_params(microbatched_model_builder))
    pipeline_history = microbatched_model.fit(microbatched_ds["train"],
                                              validation_data = microbatched_ds["val"],
                                              **fit_params)
    pipeline_result = microbatched_model.evaluate(microbatched_ds["test"], verbose = 0)
    if rank == master_rank:
      check_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_validation_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_predictions_match(reference_result, pipeline_result, num_micro_batches)