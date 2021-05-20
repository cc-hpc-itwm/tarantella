import tarantella as tnt
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched

import tarantella.strategy.pipelining.rank_mapper as rmapper
import tarantella.strategy.pipelining.core_model_builder as core_model_builder
import tarantella.strategy.pipelining.partition_generator as pgen
import models.split_layer_models as models
import utilities as utils

import pytest

import json
import logging
import numpy as np

models_and_partition_infos = [
  # Test case 0:
  # i0 --> (0) --0--> (1) --> o0
  #
  { 'model_gen' : models.fc_model_generator,
    'expected_pinfo' : models.fc_partition_info,
    'core_models' : models.fc_partitioned_core_model,
    'num_partitions' : 2
  },
  # Test case 1:
  # i0 --> (0) --0--> (1) --1--> (2) --> o0
  #
  { 'model_gen' : models.alexnet_model_generator,
    'expected_pinfo' : models.alexnet_partition_info,
    'core_models' : models.alexnet_partitioned_core_model,
    'num_partitions' : 3
  },
  # Test case 2:
  # i0 --> (0) --0--> (1) --2--> (2) --> o0
  #         |                     ^
  #         |                     |
  #         ------------1----------
  { 'model_gen' : models.skip_connection_model_generator,
    'expected_pinfo' : models.skip_connection_partition_info,
    'core_models' : models.skip_connection_partitioned_core_model,
    'num_partitions' : 3
  },
  # Test case 3:
  # i0 --> (0) --0--> (1) --2--> (2) --> o0
  #         |                    ^ ^
  #         |                    | |
  #         ------------1--------- |
  # i1 -----------------------------
  { 'model_gen' : models.multi_input_model_generator,
    'expected_pinfo' : models.multi_input_partition_info,
    'core_models' : models.multi_input_partitioned_core_model,
    'num_partitions' : 3
  },
  # Test case 4:
  # i0 --> (0) --0--> (1)----> o0
  #         |         ^  |
  #         |         |  ----> o1
  #         -----1-----
  { 'model_gen' : models.multi_output_model_generator,
    'expected_pinfo' : models.multi_output_partition_info,
    'core_models' : models.multi_output_partitioned_core_model,
    'num_partitions' : 2
  },
  # Test case 5:
  # i0 --> (0) ----> o0
  { 'model_gen' : models.simple_model_generator,
    'expected_pinfo' : models.simple_partition_info,
    'core_models' : models.simple_partitioned_core_model,
    'num_partitions' : 1
  },
]

@pytest.fixture(scope="class", params=models_and_partition_infos)
def model_and_partitions(request):
  model = request.param['model_gen']()
  partition_generator = pgen.GraphPartitionGenerator(model)
  yield model, partition_generator, request.param['num_partitions'], \
        request.param['expected_pinfo'], request.param['core_models']

@pytest.mark.min_tfversion('2.2')
class TestPartitionGenerator:

  def test_number_partitions(self, model_and_partitions):
    model, partition_gen, expected_num_partitions, _, _ = model_and_partitions
    assert partition_gen.get_number_partitions() == expected_num_partitions


  def test_partition_info(self, model_and_partitions):
    model, partition_gen, expected_num_partitions, expected_partition_gen, _ = model_and_partitions
    rank_mapper = rmapper.RankMapper(partition_gen.get_partition_graph(),
                                     nranks = expected_num_partitions)

    for rank in range(expected_num_partitions):
      partition_id = rank_mapper.get_partition_for_rank(rank)
      partition_info = pinfo.PartitionInfo(
                          partition_id = partition_id,
                          partition_graph = partition_gen.get_partition(partition_id))
      assert partition_info == expected_partition_gen(model, rank)


  def test_partition_core_models(self, model_and_partitions):
    model, partition_gen, expected_num_partitions, _, expected_model_gen = model_and_partitions
    rank_mapper = rmapper.RankMapper(partition_gen.get_partition_graph(),
                                     nranks = expected_num_partitions)

    for rank in range(expected_num_partitions):
      cm_builder = core_model_builder.CoreModelBuilder(model, partition_gen,
                                                       rank_mapper, rank)
      core_model = cm_builder.get_model()
      reference_model = expected_model_gen(rank)
      utils.check_model_configuration_identical(core_model, reference_model)
      utils.compare_weights(core_model.get_weights(), reference_model.get_weights(), 1e-6)


  @pytest.mark.parametrize("model_generator", [models.incorrect_split_model])
  def test_incorrect_split(self, model_generator):
    model = model_generator()
    with pytest.raises(RuntimeError):
      pgen.GraphPartitionGenerator(model)
