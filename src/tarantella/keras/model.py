import tarantella as tnt
import tarantella.utilities.tf_version as version_utils
from tarantella import logger

import tarantella.strategy.data_parallel.data_parallel_model as dpm
import tarantella.strategy.pipelining.partitioned_model as pm
import tarantella.strategy.pipelining.partition_generator as pgen
import tarantella.strategy.pipelining.rank_mapper as rmapper

import tensorflow as tf

# Model parallelism not supported for TF version < 2.3
TF_DEFAULT_PIPELINING_FLAG = (version_utils.tf_version_above_equal('2.3'))

class ModelMeta(type):
  def __call__(cls, *args, **kwargs):
    obj = cls._create_tnt_model(*args, **kwargs)
    return obj

  def _create_tnt_model(cls, model: tf.keras.Model,
                        parallel_strategy: tnt.ParallelStrategy = tnt.ParallelStrategy.DATA,
                        num_pipeline_stages: int = 1):
    replica_group = tnt.Group()

    if (tnt.ParallelStrategy.PIPELINING in parallel_strategy) and isinstance(model, tf.keras.Sequential):
      logger.warn(f"Cannot pipeline a `tf.keras.Sequential` model; disabling model parallelism.")
      parallel_strategy = parallel_strategy ^ tnt.ParallelStrategy.PIPELINING

    logger.info(f"Creating parallel model using {parallel_strategy}.")
    if tnt.ParallelStrategy.PIPELINING in parallel_strategy:
      rank = tnt.get_rank()

      partition_generator = pgen.GraphPartitionGenerator(model)
      rank_mapper = rmapper.RankMapper(num_ranks = tnt.get_size(),
                                       pipeline_graph = partition_generator.get_pipeline_graph())
      pipeline_group = rank_mapper.get_pipelining_group_for_rank(rank)

      logger.info(f"[Pipelining] Creating pipelined model with {pipeline_group.size} partitions.")
      # get my partition
      model = pm.PartitionedModel(model = model, group = pipeline_group,
                                  partition_generator = partition_generator, rank_mapper = rank_mapper,
                                  num_pipeline_stages=num_pipeline_stages)
      if tnt.ParallelStrategy.DATA in parallel_strategy:
        replica_group = rank_mapper.get_replica_group_for_rank(rank)
      else:
        if pipeline_group.size != tnt.get_size():
          raise ValueError(f"Provided model has only {pipeline_group.size} partitions; use {pipeline_group.size} ranks or a different parallel strategy.")

    if tnt.ParallelStrategy.DATA in parallel_strategy:
      # replicate my partition across the data parallel group
      logger.info(f"[DataParallel] Replicating local model across ranks {replica_group.group}.")
      model = dpm.DataParallelModel(model = model, group = replica_group)
    return model

class Model(metaclass = ModelMeta):
  @classmethod
  def from_config(cls, *args, **kwargs):
    # FIXME load models with any type of parallelization strategy
    logger.warning("Loading model with the default `data parallel` strategy.")
    return dpm.DataParallelModel.from_config(*args, **kwargs)


def connect_ancillary_layers(model, created_layers):
  raise AttributeError('Not supported by tarantella model. '
                       'Call `connect_ancillary_layers` on keras '
                       ' model before calling `tnt.Model()` instead.')
