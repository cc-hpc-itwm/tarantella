import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds

from tarantella import logger
import tarantella.datasets.dataset_helpers as ds_helpers
import tarantella.datasets.ops_helpers as ops_helpers
import tarantella.datasets.gradient_scaling_callback as grad_scaling
import tarantella.utilities.tf_version as version_utils

class DistributedDataset:
  def __init__(self, dataset, num_ranks, rank, shuffle_seed = 42):
    self.num_ranks = num_ranks
    self.rank = rank
    self.shuffle_seed = shuffle_seed
    self.num_samples = None

    self.base_dataset, self.dataset_transformations = \
           ops_helpers.gen_dataset_transformations(dataset)
    self.batching_info = ops_helpers.get_batching_info(self.dataset_transformations)


  def distribute_dataset_across_ranks(self, user_micro_batch_size = None, is_training = True):
    dataset = self.base_dataset

    # Batched datsets:
    # re-apply dataset transformations identically, except for batching & shuffling
    for index, (transf, ds_kwargs) in enumerate(self.dataset_transformations):
      # shuffle operation
      if isinstance(transf(dataset, **ds_kwargs), ds.ShuffleDataset):
        dataset = self.shuffle_with_seed(dataset, ds_kwargs)

      # batch operation (i.e., `batch` or `padded_batch`)
      elif self.batching_info.is_last_batching_transformation(index):
        batch_size = self.batching_info.batch_size
        if batch_size < self.num_ranks:
          raise ValueError(f"[DistributedDataset] batch size ({batch_size}) is too small " \
                           f"to be distributed to all available devices ({self.num_ranks}).")

        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          if micro_batch_size * self.num_ranks != batch_size:
            raise ValueError(f"[DistributedDataset] micro batch size ({micro_batch_size}) " \
                             f"is not consistent with batch size ({batch_size}) on the " \
                             f"number of devices used ({self.num_ranks}).")
        else:
          micro_batch_size = ds_helpers._get_microbatch_size(self.rank, self.num_ranks, batch_size)

        if is_training:
          dataset = self.distributed_batch(dataset,
                                           batch_size = batch_size,
                                           micro_batch_size = micro_batch_size)
        else:
          # FIXME: distribute batch for `evaluate` and `predict`
          dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)

      # other operations
      else:
        dataset = transf(dataset, **ds_kwargs)

    # Unbatched datasets
    if self.batching_info.is_batched == False:
      if is_training == False:    # outside `fit`
        if user_micro_batch_size:
          dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)
        else:
          dataset = self.batching_info.apply(dataset, new_batch_size = 1)

      if is_training == True:     # inside `fit`
        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          batch_size = micro_batch_size * self.num_ranks
          dataset = self.distributed_batch(dataset,
                                           batch_size = batch_size,
                                           micro_batch_size = micro_batch_size)
        else:
          raise ValueError("[DistributedDataset] Unbatched datasets without " \
                           "tnt_micro_batch_size are not supported")
    return dataset

  def shuffle_with_seed(self, dataset, ds_kwargs):
    if not 'seed' in ds_kwargs or ds_kwargs['seed'] is None:
      logger.warn("Shuffling with fixed shuffle seed {}.".format(self.shuffle_seed))
      ds_kwargs['seed'] = self.shuffle_seed
    else:
      logger.debug("Shuffling with shuffle seed {}.".format(ds_kwargs['seed']))
    return dataset.shuffle(**ds_kwargs)

  def distributed_batch(self, dataset, batch_size, micro_batch_size):
    if self.batching_info.drop_remainder == True:
      dataset = self.batching_info.apply(dataset, new_batch_size = batch_size)
      dataset = dataset.unbatch()
    else:
      if self.num_samples is None:
        self.num_samples = ds_helpers._get_num_samples(dataset)
      if self.num_samples == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("[DistributedDataset] Infinite dataset provided; cannot count samples.")

      # pad final incomplete batch to have at least `num_ranks` samples, such that
      # each rank will have the same number of iterations within one epoch
      dataset = ds_helpers._pad_dataset_if_necessary(dataset, self.num_samples, batch_size,
                                                     min_last_batch_size = self.num_ranks)

    dataset = self._get_dataset_slice_per_rank(dataset, batch_size, micro_batch_size)
    dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)

    logger.info(f"Using batch size = {batch_size}, micro batch size = {micro_batch_size}.")
    return dataset

  def _get_dataset_slice_per_rank(self, dataset, batch_size, micro_batch_size):
    if ds_helpers._is_batch_multiple_num_ranks(self.num_ranks, batch_size):
      dataset = dataset.shard(num_shards = self.num_ranks, index = self.rank)
    else:
      dataset = dataset.skip(self.rank) # skip samples up to the starting point for `rank`
      dataset = dataset.window(size = micro_batch_size,
                               shift = batch_size,
                               stride = self.num_ranks,
                               drop_remainder = False)

      kwargs = {}
      if version_utils.tf_version_above_equal('2.2'):
        kwargs['deterministic'] = True
      dataset = dataset.interleave(ds_helpers._window_datasets_to_tuples,
                                   num_parallel_calls = ds_helpers.autotune_flag(),
                                   block_length = micro_batch_size,
                                   **kwargs)
    return dataset

  def get_gradient_scaling_callback(self):
    batch_size = self.batching_info.batch_size
    scaling_factor_table = grad_scaling.build_scaling_factor_table(self.rank, self.num_ranks,
                                                                   batch_size, self.num_samples)
    if scaling_factor_table:
      return grad_scaling.ScalingFactorScheduler(scaling_factor_table)
