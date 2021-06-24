import logging
import pytest

import tensorflow as tf

import tarantella.datasets.distributed_dataset as ds
import tarantella.datasets.dataset_helpers as ds_helpers
import tarantella.utilities.tf_version as version_utils
import utilities as ds_utils

def gen_dataset_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.prefetch(buffer_size = 2)
  return dataset

def gen_dataset_shuffle_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.prefetch(buffer_size = 2)
  return dataset

def gen_dataset_padded_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.map(lambda x, y: tf.fill([4], x))
  dataset = dataset.padded_batch(batch_size,
                                 drop_remainder = drop_remainder,
                                 padded_shapes = 8)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_io_pipeline(dataset, batch_size, drop_remainder):
  # Read from multiple files in parallel
  def parse_fn(x,y):
    return x,y

  dataset = dataset.map(
      map_func = lambda x, y: parse_fn(2*x,y+1))

  dataset = dataset.cache()
  # Shuffle samples
  dataset = dataset.shuffle(1000, seed = 123)

  # Set number of samples if specified
  dataset = dataset.take(batch_size * 8)
  dataset = dataset.repeat(2)

  # Preprocess samples (in parallel)
  dataset = dataset.map(
      parse_fn, num_parallel_calls=ds_helpers.autotune_flag())

  dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
  dataset = dataset.prefetch(ds_helpers.autotune_flag())
  return dataset

def validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                           rank, comm_size):
  if version_utils.tf_version_below_equal('2.1'):
    # padding implementation not supported in TF version <= 2.1
    return ds_utils.validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                                           rank, comm_size, padded = False)
  else:
    return ds_utils.validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                                           rank, comm_size, padded = True)

transformation_test_cases = [gen_dataset_batch,
                             gen_dataset_shuffle_batch,
                             gen_dataset_io_pipeline,
                             gen_dataset_padded_batch,
                             ]
remainder_samples_per_batch = [0, # batch size is a multiple of the number of ranks
                               2, # some ranks have one additional sample in their micro-batch
                               ]
last_batch_sizes = [0, # number of samples is a multiple of batch size
                    2, # last_batch_size < number of ranks (i.e., padding is required)
                    5, # last batch is incomplete, last_batch_size >= number of ranks (no padding)
                    ]
@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [ds_utils.np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [1,3,4])
@pytest.mark.parametrize("intended_local_micro_batch_size", [5])
@pytest.mark.parametrize("num_batches", [6])
@pytest.mark.parametrize("size_last_batch", last_batch_sizes)
@pytest.mark.parametrize("size_batch_remainder", remainder_samples_per_batch)
@pytest.mark.parametrize("drop_remainder", [pytest.param(False,
                                                         marks=pytest.mark.min_tfversion('2.2')),
                                            True])
def test_micro_batching(apply_transformations, dataset_generator,
                        comm_size, intended_local_micro_batch_size, num_batches,
                        size_last_batch, size_batch_remainder, drop_remainder):
  batch_size = comm_size * intended_local_micro_batch_size + size_batch_remainder
  num_samples = num_batches * batch_size + size_last_batch
  (x_train, y_train) = dataset_generator(num_samples)

  for rank in range(comm_size):   # verify each rank separately
    reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    tf_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

    reference_dataset = apply_transformations(reference_dataset,
                                        batch_size = batch_size,
                                        drop_remainder = drop_remainder)
    tf_dataset = apply_transformations(tf_dataset,
                                       batch_size = batch_size,
                                       drop_remainder = drop_remainder)

    # load local dataset for `rank`
    tnt_dataset = ds.DistributedDataset(tf_dataset,
                                        num_ranks = comm_size,
                                        rank = rank)
    local_dataset = tnt_dataset.distribute_dataset_across_ranks()
    micro_batch_size = ds_helpers._get_microbatch_size(rank, comm_size, batch_size)

    validate_local_dataset(reference_dataset, local_dataset, micro_batch_size,
                           rank, comm_size)
