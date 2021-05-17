import numpy as np
import pytest

import tensorflow as tf

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


def _pad(array, reference):
  result = np.zeros(reference.shape)
  insertHere = [slice(0, array.shape[dim]) for dim in range(array.ndim)]
  result[tuple(insertHere)] = array
  return result

def validate_local_dataset(ref_dataset, local_dataset, micro_batch_size,
                          rank, comm_size, padded = False):
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

