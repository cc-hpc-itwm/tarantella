import pytest

import tensorflow as tf

from tarantella.datasets import distributed_dataset as ds
from tarantella.datasets import dataset_helpers as ds_helpers
import utilities as ds_utils

def gen_dataset_multiple_batch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(1, drop_remainder = True)
  dataset = dataset.batch(1, drop_remainder= True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_batch_unbatch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.repeat(2)
  dataset = dataset.batch(1)
  dataset = dataset.unbatch()
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_filter(dataset, batch_size, drop_remainder = True):
  def pred(x,y):
    return x > 5
  dataset = dataset.filter(predicate = lambda x, y: pred(x,y))
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_flat_map(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  # flat map works on batched datasets
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_flat_map_after_batch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)

  # flat map works on batched datasets
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x+6, 3*y)))
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((3*x, y)))
  return dataset

def gen_dataset_interleave(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               deterministic = True)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_interleave_after_batch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 3, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               deterministic = True)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.interleave(map_func = lambda x,y: tf.data.Dataset.from_tensors((x, y)),
                               block_length=2,
                               deterministic = True)
  return dataset

def gen_dataset_interleave_v1(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map(dataset, batch_size, drop_remainder = True):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_map_after_batch(dataset, batch_size, drop_remainder = True):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.map(lambda x, y: map_fn(x, y),
                        deterministic = True)
  return dataset

def gen_dataset_map_v1(dataset, batch_size, drop_remainder = True):
  def map_fn(x, y):
    return x*5, y
  dataset = dataset.map(lambda x, y: map_fn(x, y))
    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_interleave(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               num_parallel_calls=4,
                               deterministic = True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_interleave_after_batch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensors((2*x, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               block_length=2,
                               num_parallel_calls=4,
                               deterministic = True)
  return dataset

def gen_dataset_parallel_interleave_v1(dataset, batch_size, drop_remainder = True):
  dataset = dataset.batch(batch_size = 1, drop_remainder = False)
  dataset = dataset.interleave(map_func = lambda x, y: tf.data.Dataset.from_tensor_slices((x+3, y)),
                               cycle_length=tf.data.experimental.AUTOTUNE,
                               num_parallel_calls=4)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map(dataset, batch_size, drop_remainder = True):
  dataset = dataset.repeat(2)

  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2,
                        deterministic=True)
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_parallel_map_after_batch(dataset, batch_size, drop_remainder = True):
  dataset = dataset.repeat(2)

  def map_fn(x,y):
    return (x*5,y)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2,
                        deterministic=True)
  return dataset
    
def gen_dataset_parallel_map_v1(dataset, batch_size, drop_remainder = True):
  dataset = dataset.repeat(2)
  def map_fn(x,y):
    return x*5, y+x
  dataset = dataset.map(map_func = lambda x, y: map_fn(x,y),
                        num_parallel_calls=2)
    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_concatenate(dataset, batch_size, drop_remainder = True):
  dataset = dataset.concatenate(dataset)    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

def gen_dataset_zip(dataset, batch_size, drop_remainder = True):
  dataset = tf.data.Dataset.zip((dataset, dataset))    
  dataset = dataset.batch(batch_size, drop_remainder)
  return dataset

transformation_test_cases = [ gen_dataset_multiple_batch,
                              gen_dataset_batch_unbatch,
                              gen_dataset_filter,
                              gen_dataset_flat_map,
                              gen_dataset_flat_map_after_batch,
                              pytest.param(gen_dataset_map,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_map_after_batch,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_map_v1,
                                           marks=pytest.mark.max_tfversion('2.1')),
                              pytest.param(gen_dataset_interleave,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_interleave_after_batch,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_interleave_v1,
                                           marks=pytest.mark.max_tfversion('2.1')),
                              pytest.param(gen_dataset_parallel_interleave,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_interleave_after_batch,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_interleave_v1,
                                           marks=pytest.mark.max_tfversion('2.1')),
                              pytest.param(gen_dataset_parallel_map,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_map_after_batch,
                                           marks=pytest.mark.min_tfversion('2.2')),
                              pytest.param(gen_dataset_parallel_map_v1,
                                           marks=pytest.mark.min_tfversion('2.1')),
                              gen_dataset_concatenate,
                              gen_dataset_zip,
                              ]

@pytest.mark.parametrize("apply_transformations", transformation_test_cases)
@pytest.mark.parametrize("dataset_generator", [ds_utils.np_arrays_from_range])
@pytest.mark.parametrize("comm_size", [1,3,4])
@pytest.mark.parametrize("micro_batch_size", [7])
@pytest.mark.parametrize("num_batches", [5])
def test_batch_with_pad(apply_transformations, dataset_generator,
                        comm_size, micro_batch_size, num_batches):
  batch_size = comm_size * micro_batch_size
  num_samples = num_batches * batch_size
  (x_train, y_train) = dataset_generator(num_samples)
  
  reference_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  tnt_dataset =  tf.data.Dataset.from_tensor_slices((x_train, y_train))

  tnt_dataset = apply_transformations(tnt_dataset,
                                      batch_size = batch_size)

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
                                        batch_size = batch_size)

    ds_utils.validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                                    rank, comm_size = comm_size)
