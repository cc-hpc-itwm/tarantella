import logging
import numpy as np
import pytest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tarantella.datasets import distributed_dataset as ds
from tarantella.datasets import dataset_helpers as ds_helpers

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


def gen_dataset_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_multiple_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(1, drop_remainder = True)
  dataset = dataset.batch(1, drop_remainder= True)
    
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_shuffle_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)
  
  if pad:
    if drop_remainder:
      dataset = dataset.batch(batch_size, drop_remainder)
      dataset = dataset.unbatch()
    num_samples = ds_helpers.get_num_samples(dataset)
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_filter(dataset, batch_size, drop_remainder,comm_size,pad=False):
  # Read from multiple files in parallel
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)
  
  def pred(x,y):
    return x > 2
  dataset = dataset.filter(predicate = lambda x, y: pred(x,y))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_filter_after_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.shuffle(10, seed=44, reshuffle_each_iteration=True)

  def pred1(x,y):
    return x > 3
  dataset = dataset.filter(predicate = lambda x, y: pred1(x,y))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)

  dataset = dataset.batch(batch_size, drop_remainder)

  def pred2(x,y): # apply to batched dataset
    return x[1] > 10
  dataset = dataset.filter(predicate = lambda x, y: pred2(x,y))
  return dataset

def gen_dataset_flat_map(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)

  # flat map works on batched datasets
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_flat_map_after_batch(dataset, batch_size, drop_remainder, comm_size,pad=False):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)

  # flat map works on batched datasets
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x+6, 3*y)))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)

  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((3*x, y)))
  return dataset

def gen_dataset_interleave(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               deterministic = True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_interleave_after_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               deterministic = True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.interleave(map_func = lambda x,y: tf.data.Dataset.from_tensors((x, y)),
                               block_length=2,
                               deterministic = True)
  return dataset

def gen_dataset_interleave_v1(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map(dataset, batch_size, drop_remainder,comm_size,pad=False):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map_after_batch(dataset, batch_size, drop_remainder):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.map(lambda x, y: map_fn(x+4, y),
                        deterministic = True)
  return dataset

def gen_dataset_map_v1(dataset, batch_size, drop_remainder,comm_size,pad=False):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_padded_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.map(lambda x, y: tf.fill([4], x))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.padded_batch(batch_size,
                                 drop_remainder = drop_remainder,
                                 padded_shapes = 8)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def gen_dataset_parallel_interleave(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               num_parallel_calls=4,
                               deterministic = True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_interleave_after_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensors((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               num_parallel_calls=4,
                               deterministic = True)
  return dataset

def gen_dataset_parallel_interleave_v1(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               num_parallel_calls=4)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.repeat(2)

  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2,
                        deterministic=True)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map_after_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.repeat(2)

  def map_fn(x,y):
    return (x*5,y)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2,
                        deterministic=True)
  return dataset

def gen_data_multiple_batch(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.repeat(2)
  dataset = dataset.batch(1)
  dataset = dataset.unbatch()
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
    
  return dataset
    
def gen_dataset_parallel_map_v1(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.repeat(2)
  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_io_pipeline(dataset, batch_size, drop_remainder,comm_size,pad=False):
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
  if pad:
    if drop_remainder:
      dataset = dataset.batch(batch_size, drop_remainder)
      dataset = dataset.unbatch()
    num_samples = ds_helpers.get_num_samples(dataset)
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def gen_dataset_concatenate(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = dataset.concatenate(dataset)
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    print("call pad")
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_zip(dataset, batch_size, drop_remainder,comm_size,pad=False):
  dataset = tf.data.Dataset.zip((dataset, dataset))
  if pad:
    num_samples = ds_helpers.get_num_samples(dataset)
    if drop_remainder:
      num_samples = int(num_samples//batch_size)*batch_size
    dataset = ds_helpers.pad_dataset(dataset,batch_size,comm_size,num_samples)
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
    print("expected_batch",expected_batch)
    expected_micro_batch = expected_batch[rank * micro_batch_size:
                                          ((rank+1) * micro_batch_size)]
#     expected_micro_batch = expected_batch[rank::micro_batch_size]
    
    # this might not be true now
    assert np.array_equal(local_batch,expected_micro_batch)

  # verify that the two datasets have the same length
  with pytest.raises(StopIteration):
    next(local_dataset_it)
  with pytest.raises(StopIteration):
    next(expected_dataset_it)

transformation_test_cases = [ gen_dataset_batch,
                              gen_data_multiple_batch,
                              gen_dataset_shuffle_batch,
                              gen_dataset_multiple_batch,
                              gen_dataset_io_pipeline,
                              gen_dataset_filter,
                              gen_dataset_filter_after_batch,
                              gen_dataset_flat_map,
                              gen_dataset_flat_map_after_batch,
                              pytest.param(gen_dataset_map,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_map_after_batch,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_map_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_interleave,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_interleave_after_batch,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_interleave_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_parallel_interleave,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_interleave_after_batch,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_interleave_v1,
                                           marks=[pytest.mark.tfversion('2.0'),
                                                  pytest.mark.tfversion('2.1')]),
                              pytest.param(gen_dataset_parallel_map,
                                           marks=pytest.mark.tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_map_after_batch,
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
@pytest.mark.parametrize("num_batches", [6])
@pytest.mark.parametrize("size_final_batch", [0, 1, 6, 11])
@pytest.mark.parametrize("drop_remainder", [False,True])
def test_no_drop_remainder(apply_transformations, dataset_generator,
                           comm_size, micro_batch_size, num_batches,
                           size_final_batch,drop_remainder):
  batch_size = comm_size * micro_batch_size
  num_samples = num_batches * batch_size + size_final_batch
  (x_train, y_train) = dataset_generator(num_samples)
  
  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

  # Dataset should behve like the sequential dataset with `drop_ramainder=True`
  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder=drop_remainder,
                                      comm_size = comm_size)

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
                                        batch_size = micro_batch_size*comm_size,
                                        drop_remainder=drop_remainder,
                                        comm_size = comm_size,
                                        pad = True)
    validate_local_dataset(ref_dataset, local_dataset, micro_batch_size, rank)


@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [3, 4])
@pytest.mark.parametrize("micro_batch_size", [5])
@pytest.mark.parametrize("size_batch_remainder", [1, 7, 11])
@pytest.mark.parametrize("drop_remainder", [False,True])
def test_batch_not_multiple_num_ranks(apply_transformations, dataset_generator,
                                      comm_size, micro_batch_size,
                                      size_batch_remainder,drop_remainder):
  batch_size = comm_size * micro_batch_size + size_batch_remainder
  num_samples = 6 * batch_size
  (x_train, y_train) = dataset_generator(num_samples)

  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  real_num_samples = num_samples

  if drop_remainder:
    real_num_samples = int(num_samples//batch_size) * batch_size

  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))
  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size,
                                      drop_remainder=drop_remainder,
                                      comm_size = comm_size)

  for rank in range(comm_size):   # verify each rank separately
    dist_dataset = ds.DistributedDataset(tnt_dataset,
                                          num_ranks = comm_size,
                                          rank = rank)
    local_dataset = dist_dataset.distribute_dataset_across_ranks()
    micro_batch_size = dist_dataset.get_microbatch_size(batch_size)

    ref_dataset = apply_transformations(reference_dataset,
                                        batch_size = micro_batch_size*comm_size,
                                        drop_remainder=drop_remainder,
                                        comm_size = comm_size,
                                        pad = True)
    validate_local_dataset(ref_dataset, local_dataset, micro_batch_size, rank)
