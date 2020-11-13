import logging
import numpy as np
import pytest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tarantella.datasets import distributed_dataset as ds

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
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_multiple_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(2, drop_remainder = True)
  dataset = dataset.batch(2, drop_remainder= True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_shuffle_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)

  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_filter(dataset, batch_size, drop_remainder):
  # Read from multiple files in parallel
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)
  
  def pred(x,y):
    return x > 100
  dataset = dataset.filter(predicate = lambda x, y: pred(x,y))
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_flat_map(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)

  # flat map works on batched datasets
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_interleave(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               deterministic = True)

  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_interleave_v1(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map(dataset, batch_size, drop_remainder):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map_v1(dataset, batch_size, drop_remainder):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y))
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_padded_batch(dataset, batch_size, drop_remainder):
  dataset = dataset.map(lambda x, y: tf.fill([4], x))
  dataset = dataset.padded_batch(batch_size,
                                 drop_remainder = drop_remainder,
                                 padded_shapes = 8)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_parallel_interleave(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               num_parallel_calls=4,
                               deterministic = True)

  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_interleave_v1(dataset, batch_size, drop_remainder):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               num_parallel_calls=4)

  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map(dataset, batch_size, drop_remainder):
  dataset = dataset.repeat(2)

  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2,
                        deterministic=True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map_v1(dataset, batch_size, drop_remainder):
  dataset = dataset.repeat(2)
  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_io_pipeline(dataset, batch_size, drop_remainder):
  # Read from multiple files in parallel
  def parse_fn(x,y):
    return x,y

  dataset = dataset.map(
      map_func = lambda x, y: parse_fn(x,y))

  dataset = dataset.cache()
  # Shuffle samples
  dataset = dataset.shuffle(1000, seed = 123)
  dataset = dataset.repeat(2)

  # Set number of samples if specified
  dataset = dataset.take(batch_size * 3)

  # Preprocess samples (in parallel)
  dataset = dataset.map(
      parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def gen_dataset_concatenate(dataset, batch_size, drop_remainder):
  dataset = dataset.concatenate(dataset)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_zip(dataset, batch_size, drop_remainder):
  dataset = tf.data.Dataset.zip((dataset, dataset))
  dataset = dataset.batch(batch_size, drop_remainder)

  return dataset

def validate_local_dataset(ref_dataset, local_dataset, micro_batch_size, rank):
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
    expected_micro_batch = expected_batch[rank * micro_batch_size:
                                          ((rank+1) * micro_batch_size)]
    assert np.array_equal(local_batch,expected_micro_batch)

  # verify that the two datasets have the same length
  with pytest.raises(StopIteration):
    next(local_dataset_it)
  with pytest.raises(StopIteration):
    next(expected_dataset_it)

transformation_test_cases = [ gen_dataset_batch,
                              gen_dataset_shuffle_batch,
                              gen_dataset_multiple_batch,
                              gen_dataset_io_pipeline,
                              gen_dataset_filter,
                              gen_dataset_flat_map,
                              pytest.param(gen_dataset_map,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_map_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_interleave,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_interleave_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_parallel_interleave,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_interleave_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_parallel_map,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_map_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              gen_dataset_padded_batch,
                              gen_dataset_concatenate,
                              gen_dataset_zip,
                              ]
@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [1,3,4])
@pytest.mark.parametrize("micro_batch_size", [5])
@pytest.mark.parametrize("num_samples", [91])
@pytest.mark.parametrize("nepochs", [2])
def test_with_drop_remainder(apply_transformations, dataset_generator,
                             comm_size, micro_batch_size, num_samples,
                             nepochs):
  batch_size = comm_size * micro_batch_size
  (x_train, y_train) = dataset_generator(num_samples)

  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder=True)

  for rank in range(comm_size):   # verify each rank separately
    # load local dataset for `rank`
    dist_dataset = ds.DistributedDataset(tnt_dataset,
                                          num_ranks = comm_size,
                                          rank = rank)
    local_dataset = dist_dataset.distribute_dataset_across_ranks()
    micro_batch_size = dist_dataset.get_microbatch_size(batch_size)

    # rebuild reference dataset each time to prevent
    # shuffling effects for repeated iterations
    ref_dataset = apply_transformations(reference_dataset,
                                        batch_size = batch_size,
                                        drop_remainder=True)
    for epoch in range(nepochs):
      validate_local_dataset(ref_dataset, local_dataset, micro_batch_size, rank)


@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [1,3,4])
@pytest.mark.parametrize("micro_batch_size", [5])
@pytest.mark.parametrize("num_batches", [4])
@pytest.mark.parametrize("size_final_batch", [0, 1, 6, 11])
def test_no_drop_remainder(apply_transformations, dataset_generator,
                           comm_size, micro_batch_size, num_batches,
                           size_final_batch):
  batch_size = comm_size * micro_batch_size
  num_samples = num_batches * batch_size + size_final_batch
  (x_train, y_train) = dataset_generator(num_samples)

  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

  # Dataset should behve like the sequential dataset with `drop_ramainder=True`
  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder=False)

  for rank in range(comm_size):   # verify each rank separately
    # load local dataset for `rank`
    dist_dataset = ds.DistributedDataset(tnt_dataset,
                                          num_ranks = comm_size,
                                          rank = rank)
    local_dataset = dist_dataset.distribute_dataset_across_ranks()
    micro_batch_size = dist_dataset.get_microbatch_size(batch_size)

    # rebuild reference dataset each time to prevent
    # shuffling effects for repeated iterations
    ref_dataset = apply_transformations(reference_dataset,
                                        batch_size = batch_size,
                                        drop_remainder=True)
    validate_local_dataset(ref_dataset, local_dataset, micro_batch_size, rank)


@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [3, 4])
@pytest.mark.parametrize("micro_batch_size", [5])
@pytest.mark.parametrize("size_batch_remainder", [1, 7, 11])
def test_batch_not_multiple_num_ranks(apply_transformations, dataset_generator,
                                      comm_size, micro_batch_size,
                                      size_batch_remainder):
  batch_size = comm_size * micro_batch_size + size_batch_remainder
  num_samples = 4 * batch_size
  (x_train, y_train) = dataset_generator(num_samples)

  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))
  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder=True)

  for rank in range(comm_size):   # verify each rank separately
    dist_dataset = ds.DistributedDataset(tnt_dataset,
                                          num_ranks = comm_size,
                                          rank = rank)
    # distributing the dataset should fail because the batch size is not a
    # multiple of the number of ranks
    with pytest.raises(ValueError):
      local_dataset = dist_dataset.distribute_dataset_across_ranks()
