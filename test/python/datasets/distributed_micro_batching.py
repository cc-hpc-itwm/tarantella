import logging
import numpy as np
import pytest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tarantella.datasets import distributed_dataset as ds
from tarantella.datasets import dataset_helpers as ds_helpers
import tarantella.utilities.tf_version as version_utils

def mnist_as_np_arrays(training_samples):
  mnist_train_size = 60000
  assert(training_samples <= mnist_train_size)

  # load given number of samples
  (x_train_all, y_train_all), _ = keras.datasets.mnist.load_data()
  x_train = x_train_all[:training_samples]
  y_train = y_train_all[:training_samples]

  # normalization and reshape
  x_train = x_train.reshape(training_samples, 28, 28, 1).astype('float32') / 255.
  y_train = y_train.astype('float32')
  return (x_train, y_train)

def np_arrays_from_range(training_samples):
  return (tf.range(training_samples), tf.range(training_samples))

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
      parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def _pad(array, reference):
  result = np.zeros(reference.shape)
  insertHere = [slice(0, array.shape[dim]) for dim in range(array.ndim)]
  result[tuple(insertHere)] = array
  return result

def _validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                            rank, comm_size, padded):
  local_dataset_it = iter(local_dataset)
  expected_dataset_it = iter(ref_dataset)

  for local_batch, expected_batch in zip(local_dataset_it, expected_dataset_it):
    # look at the first dataset when datasets are nested (e.g., after zip, or (samples, targets))
    # TODO: check all elements of the tuples
    while isinstance(local_batch, tuple):
      local_batch = local_batch[0]

    while isinstance(expected_batch, tuple):
      expected_batch = expected_batch[0]
    # extract the slice of the reference dataset that corresponds to `rank`
    expected_micro_batch = expected_batch[rank::comm_size]

    if padded: # pad the expected micro_batch to the shape of the `rank` micro_batch
               # to enable comparison
      shape_expect = np.shape(expected_micro_batch)
      shape_local = np.shape(local_batch)

      if shape_expect != shape_local:
        expected_micro_batch = _pad(expected_micro_batch, local_batch)

    assert np.array_equal(local_batch,expected_micro_batch)

  # verify that the two datasets have the same length
  with pytest.raises(StopIteration):
    next(local_dataset_it)
  with pytest.raises(StopIteration):
    next(expected_dataset_it)

def validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                           rank, comm_size):
  if version_utils.tf_version_below_equal('2.1'):
    # padding implementation not supported in TF version <= 2.1
    return _validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                                   rank, comm_size, padded = False)
  else:
    return _validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
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
@pytest.mark.parametrize("dataset_generator", [np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [1,3,4])
@pytest.mark.parametrize("micro_batch_size", [5])
@pytest.mark.parametrize("num_batches", [6])
@pytest.mark.parametrize("size_final_batch", last_batch_sizes)
@pytest.mark.parametrize("size_batch_remainder", remainder_samples_per_batch)
@pytest.mark.parametrize("drop_remainder", [False, True])
def test_micro_batching(apply_transformations, dataset_generator,
                        comm_size, micro_batch_size, num_batches,
                        size_final_batch, size_batch_remainder, drop_remainder):
  batch_size = comm_size * micro_batch_size + size_batch_remainder
  num_samples = num_batches * batch_size + size_final_batch
  (x_train, y_train) = dataset_generator(num_samples)

  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder = drop_remainder)

  for rank in range(comm_size):   # verify each rank separately
    # load local dataset for `rank`
    dist_dataset = ds.DistributedDataset(tnt_dataset,
                                          num_ranks = comm_size,
                                          rank = rank)
    local_dataset = dist_dataset.distribute_dataset_across_ranks()
    micro_batch_size = ds_helpers._get_microbatch_size(rank, comm_size, batch_size)

    # rebuild reference dataset each time to prevent
    # shuffling effects for repeated iterations
    ref_dataset = apply_transformations(reference_dataset,
                                        batch_size = batch_size,
                                        drop_remainder = drop_remainder)

    validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                           rank, comm_size)
