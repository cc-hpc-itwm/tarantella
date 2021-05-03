import tarantella as tnt
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as pipelining
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.shared_model_builder as shared
import tarantella.strategy.pipelining.microbatched_model_builder as microbatched

import tarantella.strategy.pipelining.rank_mapper as rmapper
import tarantella.strategy.pipelining.core_model_builder as core_model_builder
import tarantella.strategy.pipelining.partition_generator as pgen
import models.split_layer_models as models

import pytest

import logging
import numpy as np

models_and_partition_infos = [
  { 'model_gen' : models.fc_model_generator,
    'expected_pinfo' : models.fc_partition_info,
    'num_partitions' : 2
  },
  { 'model_gen' : models.alexnet_model_generator,
    'expected_pinfo' : models.alexnet_partition_info,
    'num_partitions' : 3
  },
  { 'model_gen' : models.skip_connection_model_generator,
    'expected_pinfo' : models.skip_connection_partition_info,
    'num_partitions' : 3
  },
  { 'model_gen' : models.multi_input_model_generator,
    'expected_pinfo' : models.multi_input_partition_info,
    'num_partitions' : 3
  },
]

@pytest.mark.tfversion(['2.2', '2.3', '2.4'])
class TestPartitionGenerator:

  @pytest.mark.parametrize("model_and_pinfo", models_and_partition_infos)
  def test_partition_info(self, model_and_pinfo):

    model = model_and_pinfo['model_gen']()
    partition_generator = pgen.GraphPartitionGenerator(model)

    rank_mapper = rmapper.RankMapper(partition_generator.get_partition_graph(),
                                     model_and_pinfo['num_partitions'])

    for rank in range(model_and_pinfo['num_partitions']):
      partition_id = rank_mapper.get_partition_for_rank(rank)
      partition_info = pinfo.PartitionInfo(
                          partition_id = partition_id,
                          partition_graph = partition_generator.get_partition(partition_id))
      assert partition_info == model_and_pinfo['expected_pinfo'](model, rank)
