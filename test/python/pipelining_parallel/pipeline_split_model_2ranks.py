import tarantella as tnt
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched

import tarantella.strategy.pipelining.rank_mapper as rmapper
import tarantella.strategy.pipelining.core_model_builder as cm_builder
import tarantella.strategy.pipelining.partition_generator as pgen
import tarantella.keras.layers as tnt_layers

import models.mnist_models as mnist
import utilities as util
from models.hardcoded_model import get_reference_compile_params
from models.hardcoded_model import get_microbatched_compile_params
from models.hardcoded_model import check_histories_match
from models.hardcoded_model import check_predictions_match
from models.hardcoded_model import check_validation_histories_match
from models.hardcoded_model import load_microbatched_datasets
from models.hardcoded_model import load_reference_datasets

import tensorflow as tf
import tensorflow.keras as keras
import pytest

import logging
import numpy as np

def simple_model_generator():
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Flatten()(input0)
  x = keras.layers.Dense(2, activation='relu')(x)
  x = tnt_layers.SplitLayer(name="split_layer1")(x)
  output = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  model = keras.Model(inputs=input0, outputs=output)
  return model

def to_microbatched(model, micro_batch_size, num_micro_batches, num_batches, num_test_batches):
  rank = tnt.get_rank()
  partition_generator = pgen.GraphPartitionGenerator(model)
  rank_mapper = rmapper.RankMapper(num_ranks = tnt.get_size(),
                                   pipeline_graph = partition_generator.get_pipeline_graph())

  partition_id = rank_mapper.get_partition_for_rank(rank)
  partition_graph = partition_generator.get_partition_graph(partition_id)
  partition_info = pinfo.PartitionInfo(partition_id = partition_id,
                                       partition_graph = partition_graph)

  core_model_builder = cm_builder.CoreModelBuilder(model, partition_id, partition_graph)
  core_model = core_model_builder.get_model()

  connection_table = rank_mapper.get_connections_for_rank(rank)
  pipeline_communicator = tnt.PipelineCommunicator(connection_table, num_micro_batches)

  shared_model_builder = shared.SharedModelBuilder(partition_info, core_model,
                                                   pipeline_communicator, micro_batch_size)
  shared_model = shared_model_builder.get_model()

  microbatched_model_builder = microbatched.MicrobatchedModelBuilder(partition_info, shared_model,
                                                                     micro_batch_size, num_micro_batches)
  ds = load_microbatched_datasets(micro_batch_size, num_micro_batches,
                                  num_batches, num_test_batches, partition_info)
  pipeline_communicator.setup_infrastructure(micro_batch_size)
  return microbatched_model_builder, ds

@pytest.mark.min_tfversion('2.2')
class TestPipeline_SplitPartitions_AutoMicrobatching:

  @pytest.mark.parametrize("model_generator", [simple_model_generator])
  @pytest.mark.parametrize("num_micro_batches", [2])
  @pytest.mark.parametrize("micro_batch_size", [16])
  @pytest.mark.parametrize("num_batches", [10])
  @pytest.mark.parametrize("num_test_batches", [10])
  @pytest.mark.parametrize("number_epochs", [3])
  def test_train(self, model_generator, num_micro_batches, micro_batch_size,
                 num_batches, num_test_batches, number_epochs):
    batch_size = micro_batch_size * num_micro_batches
    fit_params = {'epochs' : number_epochs, 'shuffle' : False, 'verbose' : 0}
    rank = tnt.get_rank()
    master_rank = tnt.get_size() - 1  # the last partition will be assigned to rank (nranks-1)

    # reference model
    if rank == master_rank:
      reference_ds = load_reference_datasets(batch_size, num_batches, num_test_batches)
      reference_model = model_generator()

      reference_model.compile(**get_reference_compile_params())
      reference_history = reference_model.fit(reference_ds["train"],
                                              validation_data = reference_ds["val"],
                                              **fit_params)
      reference_result = reference_model.evaluate(reference_ds["test"], verbose = 0)

    # pipelined model
    model = model_generator()
    microbatched_model_builder, microbatched_ds = to_microbatched(model, micro_batch_size,
                                                  num_micro_batches, num_batches, num_test_batches)
    microbatched_model = microbatched_model_builder.get_model()
    microbatched_model.summary()

    microbatched_model.compile(**get_microbatched_compile_params(microbatched_model_builder))
    pipeline_history = microbatched_model.fit(microbatched_ds["train"],
                                              validation_data = microbatched_ds["val"],
                                              **fit_params)
    pipeline_result = microbatched_model.evaluate(microbatched_ds["test"], verbose = 0)

    if rank == master_rank:
      print (reference_history.history)
      print (pipeline_history.history)
      check_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_validation_histories_match(reference_history, pipeline_history, num_micro_batches)
      check_predictions_match(reference_result, pipeline_result, num_micro_batches)

