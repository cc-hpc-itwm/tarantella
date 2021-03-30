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
from hardcoded_model import *

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
import pytest

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
    pipeline_communicator = get_pipeline_communicator(micro_batch_size = micro_batch_size,
                                                      num_micro_batches = num_micro_batches)
    core_model = get_partitioned_core_model()
    
    ### LOAD DATASETS
    (x_train, y_train), _, _ = mnist.load_mnist_dataset(train_size, 0, 0)
    train_dataset_reference = util.create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                              .shuffle(len(x_train), shuffle_seed)

    partition_train_dataset = get_microbatched_dataset(x_train, y_train,
                                                       micro_batch_size, num_micro_batches,
                                                       core_model) \
                              .shuffle(len(x_train), shuffle_seed)

    if rank == p_0_rank:
      losses = {}
      metrics = {}
    elif rank == p_1_rank:
      losses = {0 : keras.losses.SparseCategoricalCrossentropy()}
      metrics = {0 : keras.metrics.SparseCategoricalAccuracy()}

    partition_info = get_partition_info(core_model)
    shared_model_builder = shared.SharedModelBuilder(partition_info, core_model, pipeline_communicator, micro_batch_size)
    shared_model = shared_model_builder.get_model()

    microbatched_model_builder = microbatched.MicrobatchedModelBuilder(partition_info, shared_model,
                                                                       micro_batch_size, num_micro_batches)
    microbatched_model = microbatched_model_builder.get_model()

    microbatched_model.compile(optimizer = keras.optimizers.SGD(learning_rate),
                               loss = microbatched_model_builder.get_losses(losses),
                               loss_weights = microbatched_model_builder.get_loss_weights(),
                               metrics = microbatched_model_builder.get_metrics(metrics))
    # keras.utils.plot_model(core_model, f"partition_{partition_info.partition_id}_core.png", show_shapes=True)
    # keras.utils.plot_model(shared_model, f"partition_{partition_info.partition_id}_shared.png", show_shapes=True)
    # keras.utils.plot_model(microbatched_model, f"partition_{partition_info.partition_id}_microbatched.png", show_shapes=True)
