import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds

from tarantella import logger
import tarantella.datasets.dataset_helpers as ds_helpers
import tarantella.utilities.tf_version as version_utils

def _get_microbatch_size(rank, num_ranks, batch_size):
  if batch_size is None or batch_size == 0:
    raise ValueError("[DistributedDataset]Incorrectly defined batch size")

  microbatch_size = int(batch_size // num_ranks)
  remaining_samples = batch_size % num_ranks

  if remaining_samples != 0:
    logger.debug(f"Batch size ({batch_size}) is a not multiple of the number of ranks {num_ranks}.")
  if rank < remaining_samples:
    microbatch_size = microbatch_size + 1

  logger.debug(f"Rank {rank} has micro batch {microbatch_size}.")
  return microbatch_size

def _is_batch_multiple_num_ranks(num_ranks, batch_size):
  return batch_size % num_ranks == 0

def _is_num_samples_multiple_batch_size(num_samples, batch_size):
  return num_samples % batch_size == 0

def _get_last_incomplete_batch_size(num_samples, batch_size):
  return num_samples - int(num_samples // batch_size) * batch_size

def _window_datasets_to_tuples(*datasets_in_window):
  # Datasets in a window
  # win0 = [ (i0_0, i1_0, label0_0),  # each sample is a tuple
  #          (i0_1, i1_1, label0_1),
  #          (i0_2, i1_2, label0_2),]
  # Returns:
  # batched_datasets = ( [i0_0, i0_1, i0_2],# each dataset in the tuple has `window_size` samples
  #                      [i1_0, i1_1, i1_2],
  #                      [label0_0, label0_1, label0_2] )
  batched_datasets = list()

  for datasets in datasets_in_window:  # one dataset tuple for each sample in the window
    if not isinstance(datasets,tuple):
      datasets = [datasets]

    temp = []
    for dataset in datasets:
      temp.append(dataset)
    batched_datasets.append(tuple(temp))

  if len(batched_datasets) == 1:
    return tf.data.Dataset.from_generator(batched_datasets[0])
  return tf.data.Dataset.zip(tuple(batched_datasets))

def _get_scaling_factor(micro_batch_size, batch_size, num_ranks):
  return micro_batch_size * num_ranks / batch_size

def _get_scaling_factor_by_iteration(iteration_id, scaling_factor_table):
  scaling_factor = 1.0
  for min_iteration_id, value in sorted(scaling_factor_table.items()):
    if iteration_id >= min_iteration_id:
      scaling_factor = value
  return scaling_factor

def _build_scaling_factor_table(rank, num_ranks, num_samples, batch_size):
  # Defines the gradient `scaling_factor` to be used for each iteration starting
  # with `start_iteration_id`
  # scaling_factor_table = { start_iteration_id: scaling_factor }

  if _is_batch_multiple_num_ranks(num_ranks, batch_size) and \
      _is_num_samples_multiple_batch_size(num_samples, batch_size):
    return None

  micro_batch_size = _get_microbatch_size(rank, num_ranks, batch_size)

  # each iteration starting with id 0 will use a scaling factor defined by
  # the rank's micro batch size
  scaling_factor_table = { 0 : _get_scaling_factor(micro_batch_size, batch_size, num_ranks) }

  # the last iteration (with an incomplete batch) uses a separate scaling factor
  if not _is_num_samples_multiple_batch_size(num_samples, batch_size):
    final_iteration_id = int(num_samples // batch_size)

    last_batch_size = _get_last_incomplete_batch_size(num_samples, batch_size)
    last_micro_batch_size = _get_microbatch_size(rank, num_ranks, last_batch_size)
    last_iteration_scaling_factor = _get_scaling_factor(last_micro_batch_size,
                                                        last_batch_size, num_ranks)

    scaling_factor_table[final_iteration_id] = last_iteration_scaling_factor
  return scaling_factor_table

class DistributedDataset:
  def __init__(self, dataset, num_ranks, rank, shuffle_seed = 42):
    self.num_ranks = num_ranks
    self.rank = rank
    self.shuffle_seed = shuffle_seed

    self.dataset = dataset
    self.base_dataset, self.dataset_transformations = \
           ds_helpers.gen_dataset_transformations(dataset)
    self.batching_info = ds_helpers.get_batching_info(self.dataset_transformations)


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
        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          if micro_batch_size * self.num_ranks != batch_size:
            raise ValueError(f"[DistributedDataset] micro batch size ({micro_batch_size}) " \
                             f"is not consistent with batch size ({batch_size}) on the " \
                             f"number of devices used ({num_ranks}).")
        else:
          micro_batch_size = _get_microbatch_size(self.rank, self.num_ranks, batch_size)

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

  def pad_dataset(self, dataset, batch_size, num_samples):
    real_batch_size = batch_size
    
    last_batch_size = _get_last_incomplete_batch_size(num_samples, batch_size)
    if last_batch_size == 0:
      logger.debug(f"No padding required: number of samples {num_samples} is a multiple " \
                   f"of the batch size {batch_size}.")
      return dataset
    
    if version_utils.tf_version_below_equal('2.1'):
      num_samples_multiple = num_samples - last_batch_size
      logger.warn(f"Number of samples ({num_samples}) is not a multiple of batch size. " \
                  f"Last batch padding not supported in TF v{version_utils.current_version()}. " \
                  f"Dropping the last incomplete batch from the dataset. "\
                  f"Now dataset has {num_samples_multiple} samples.")
      return dataset.take(num_samples_multiple)

    logger.debug(f"Incomplete last batch in the dataset: number of samples is " \
                 f"{last_batch_size} ( != batch size {batch_size}).")

    if last_batch_size < self.num_ranks:
      logger.debug(f"Padding required for the last batch: number of samples is " \
                   f"{last_batch_size} ( < nranks {self.num_ranks}).")

      num_padded = self.num_ranks - last_batch_size
      rest_dataset = dataset.take(2*self.num_ranks - num_padded)
      logger.info("Dataset is padded with {} elements.".format(
              num_padded))
      rest_dataset = rest_dataset.batch(self.num_ranks,drop_remainder=False)

      # If padded_shape is unset, all dimensions of all components are padded to the maximum size
      # in the batch.
      rest_dataset = rest_dataset.padded_batch(2)
      rest_dataset = rest_dataset.unbatch()
      rest_dataset = rest_dataset.unbatch()

      rest_dataset = rest_dataset.skip(2*self.num_ranks - num_padded)
      dataset = dataset.concatenate(rest_dataset)
    return dataset

  def distributed_batch(self, dataset, batch_size, micro_batch_size):
    if self.batching_info.drop_remainder == True:
      dataset = self.batching_info.apply(dataset, new_batch_size = batch_size)
      dataset = dataset.unbatch()
    else:
      num_samples = ds_helpers.get_num_samples(dataset)
      self.num_samples = num_samples
      if num_samples == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("[DistributedDataset] Infinite dataset provided; cannot count samples.")
      dataset = self.pad_dataset(dataset, batch_size, num_samples)

    dataset = self._get_dataset_slice_per_rank(dataset, batch_size, micro_batch_size)
    dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)

    logger.info("Using batch size = {batch_size}, micro batch size = {micro_batch_size}.")
    return dataset

  def _get_dataset_slice_per_rank(self, dataset, batch_size, micro_batch_size):
    if _is_batch_multiple_num_ranks(self.num_ranks, batch_size):
      dataset = dataset.shard(num_shards = self.num_ranks, index = self.rank)
    else:
      dataset = dataset.skip(self.rank) # skip samples up to the starting point for `rank`
      dataset = dataset.window(size = micro_batch_size,
                               shift = batch_size,
                               stride = self.num_ranks,
                               drop_remainder = False)
      dataset = dataset.interleave(_window_datasets_to_tuples,
                                   num_parallel_calls = tf.data.AUTOTUNE,
                                   block_length = micro_batch_size,
                                   deterministic = True)
    return dataset

  def generate_callback_if_have(self):
    batch_size = self.batching_info.batch_size
    scaling_factor_table = _build_scaling_factor_table(self.rank, self.num_ranks,
                                                       self.num_samples, batch_size)
    if scaling_factor_table:
      return ds_helpers.ScalingFactorScheduler(scaling_factor_table,
                                               _get_scaling_factor_by_iteration)
