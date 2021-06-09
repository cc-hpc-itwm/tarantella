import tensorflow as tf

import tarantella as tnt
from tarantella import logger
import tarantella.utilities.tf_version as version_utils

def _get_num_samples(dataset):
  cardinality = tf.data.experimental.cardinality(dataset)

  if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
    logger.debug("Infinite dataset detected.")
    return tf.data.experimental.INFINITE_CARDINALITY

  if cardinality != tf.data.experimental.UNKNOWN_CARDINALITY:
    logger.debug("Dataset size is %d" % (cardinality.numpy()))
    return cardinality.numpy()

  logger.debug("Unknown dataset size. Counting samples...")
  dataset_size = 0
  for d in dataset:
    dataset_size += 1
  logger.debug("Dataset size is %d" % (dataset_size))
  return dataset_size

def _get_microbatch_size(rank, num_ranks, batch_size):
  if batch_size is None or batch_size == 0:
    raise ValueError("[DistributedDataset]Incorrectly defined batch size")

  microbatch_size = int(batch_size // num_ranks)
  remaining_samples = batch_size % num_ranks

  if remaining_samples != 0:
    logger.debug(f"[Rank {tnt.get_rank()}] Batch size ({batch_size}) is a not multiple of the number of ranks {num_ranks}.")
  if rank < remaining_samples:
    microbatch_size = microbatch_size + 1

  logger.debug(f"[Rank {tnt.get_rank()}] Micro batch size {microbatch_size}.")
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
  datasets_list = list()
  for datasets in datasets_in_window:  # one dataset tuple for each sample in the window
    if not isinstance(datasets,tuple):
      datasets = [datasets]
    datasets_list.append(tuple(list(datasets)))

  return tf.data.Dataset.zip(tuple(datasets_list))

def _pad_dataset_if_necessary(dataset, num_samples, batch_size, min_batch_size):
  last_batch_size = _get_last_incomplete_batch_size(num_samples, batch_size)
  if last_batch_size == 0:
    logger.debug(f"No padding required: number of samples {num_samples} is a multiple " \
                 f"of the batch size {batch_size}.")
    return dataset

  logger.info(f"Incomplete last batch in the dataset: number of samples is " \
              f"{last_batch_size} ( != batch size {batch_size}).")

  if version_utils.tf_version_below_equal('2.1'):
    num_samples_multiple = num_samples - last_batch_size
    logger.warn(f"Number of samples ({num_samples}) is not a multiple of batch size. " \
                f"This use case is not supported in TF v{version_utils.current_version()}. " \
                f"Dropping the last incomplete batch from the dataset, "\
                f"and proceeding with {num_samples_multiple} samples.")
    return dataset.take(num_samples_multiple)

  if last_batch_size < min_batch_size:
    logger.debug(f"Padding required for the last batch: number of samples is " \
                 f"{last_batch_size} ( < min_batch_size {min_batch_size}).")

    # Create helper dataset that contains one full batch and one incomplete batch
    helper_dataset = dataset.take(min_batch_size + last_batch_size)
    helper_dataset = helper_dataset.batch(min_batch_size, drop_remainder=False)

    # If `padded_shape` is unspecified, all dimensions of all components
    # are padded to the maximum size in the batch.
    # The second batch in `helper_dataset` will now contain `min_batch_size - last_batch_size`
    # default-initialized samples.
    helper_dataset = helper_dataset.padded_batch(2)

    # Switch back to a list of samples instead of batches
    helper_dataset = helper_dataset.unbatch().unbatch()

    # Remaining samples in the dataset are those generated through padding
    padding_samples = helper_dataset.skip(min_batch_size + last_batch_size)
    dataset = dataset.concatenate(padding_samples)
    logger.info(f"[Rank {tnt.get_rank()}] Dataset padded with " \
                f"{min_batch_size - last_batch_size} samples.")
  return dataset

def autotune_flag():
  if version_utils.tf_version_below_equal('2.3'):
    return tf.data.experimental.AUTOTUNE
  else:
    return tf.data.AUTOTUNE

