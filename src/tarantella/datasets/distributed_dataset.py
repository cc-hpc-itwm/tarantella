import logging
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds

import tarantella.datasets.dataset_helpers as ds_helpers

class DistributedDataset:
  def __init__(self, dataset, num_ranks, rank, shuffle_seed = 42):
    self.num_ranks = num_ranks
    self.rank = rank
    self.shuffle_seed = shuffle_seed

    self.dataset = dataset
    self.base_dataset, self.dataset_transformations = \
           ds_helpers.gen_dataset_transformations(dataset)

  def distribute_dataset_across_ranks(self, user_micro_batch_size = None, is_training = True):
    index_last_batch_op = ds_helpers.get_index_last_batch_operation(self.dataset_transformations)
    dataset = self.base_dataset

    # Batched datsets:
    # re-apply dataset transformations identically, except for batching & shuffling
    for index, (transf, ds_kwargs) in enumerate(self.dataset_transformations):
      # shuffle operation
      if isinstance(transf(dataset, **ds_kwargs), ds.ShuffleDataset):
        dataset = self.shuffle_with_seed(dataset, ds_kwargs)
      # batch operation
      elif isinstance(transf(dataset, **ds_kwargs), ds.BatchDataset) \
      and index_last_batch_op == index:
        batch_size = self.get_batch_size(ds_kwargs)
        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          if micro_batch_size * self.num_ranks != batch_size:
            raise ValueError("[DistributedDataset] micro batch size (%d) is not consistent \
with batch size (%d) on number of devices used (%d)" % (micro_batch_size, batch_size, self.num_ranks))
        else:
          micro_batch_size = self.get_microbatch_size(batch_size)

        drop_remainder = False
        if 'drop_remainder' in ds_kwargs:
          drop_remainder = ds_kwargs['drop_remainder']

        if is_training:
          dataset = self.distributed_batch(dataset,
                                           batch_size = batch_size,
                                           micro_batch_size = micro_batch_size,
                                           drop_remainder = drop_remainder)
        else:
          # FIXME: distribute batch for `evaluate` and `predict`
          dataset = dataset.batch(batch_size = micro_batch_size,
                                  drop_remainder = drop_remainder)
      # other operations
      else:
        dataset = transf(dataset, **ds_kwargs)

    # Unbatched datasets outside `fit`
    if is_training == False and index_last_batch_op == None:
      if user_micro_batch_size:
        dataset = dataset.batch(batch_size = user_micro_batch_size)
      else:
        dataset = dataset.batch(batch_size = 1)

    # Unbatched datasets inside `fit`
    if is_training == True and index_last_batch_op == None:
      if user_micro_batch_size:
        micro_batch_size = user_micro_batch_size
        batch_size = micro_batch_size * self.num_ranks
        dataset = self.distributed_batch(dataset,
                                         batch_size = batch_size,
                                         micro_batch_size = micro_batch_size,
                                         drop_remainder = False)
      else:
        raise ValueError("[DistributedDataset] Unbatched datasets without tnt_micro_batch_size are not supported")

    return dataset

  def shuffle_with_seed(self, dataset, ds_kwargs):
    if not 'seed' in ds_kwargs or ds_kwargs['seed'] is None:
      logging.getLogger().warn("[rank %d] Shuffling with fixed shuffle seed %d" % (
                              self.rank, self.shuffle_seed))
      ds_kwargs['seed'] = self.shuffle_seed
    else:
      logging.getLogger().debug("[rank %d] Shuffling with shuffle seed %d" % (
                                self.rank, ds_kwargs['seed']))
    return dataset.shuffle(**ds_kwargs)

  def distributed_batch(self, dataset, batch_size, micro_batch_size, drop_remainder):
    if drop_remainder == True:
      dataset = dataset.batch(batch_size = batch_size,
                              drop_remainder = True)
      dataset = dataset.unbatch()

    else: # no drop remainder
      num_samples = ds_helpers.get_num_samples(dataset)
      if num_samples == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("[DistributedDataset] Infinite dataset provided")

      # Total number of samples is not multiple of the batch size
      if num_samples % batch_size != 0:
        logging.getLogger().warn("[rank %d] Number of samples (%d) is not a multiple of batch size.\
 Removing the last incomplete batch from the dataset." % (self.rank, num_samples))
        num_samples_multiple = (num_samples // batch_size) * batch_size
        dataset = dataset.take(num_samples_multiple)

    dataset = dataset.batch(batch_size = micro_batch_size,
                            drop_remainder = False)
    dataset = dataset.shard(num_shards=self.num_ranks, index = self.rank)

    logging.getLogger().info("[rank %d] Using batch size = %d, micro batch size = %d." \
                             % (self.rank, batch_size, micro_batch_size))
    return dataset

  def get_batch_size(self, ds_kwargs):
    if not 'batch_size' in ds_kwargs:
      raise KeyError("[DistributedDataset] Batch transformation defined without batch size")
    return ds_kwargs['batch_size']

  def get_microbatch_size(self, batch_size):
    if batch_size is None or batch_size == 0:
      raise ValueError("[DistributedDataset]Incorrectly defined batch size")

    if batch_size % self.num_ranks != 0:
      raise ValueError("[DistributedDataset] Batch size (%d) is not a multiple of the number of ranks %d" % (
                              batch_size, self.num_ranks))

    logging.getLogger().debug("[rank %d] Batch size (%d) is a multiple of the number of ranks %d" % (
                              self.rank, batch_size, self.num_ranks))
    return int(batch_size // self.num_ranks)
