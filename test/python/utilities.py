import datetime
import tensorflow as tf
import numpy as np
import logging

def create_dataset_from_arrays(samples, labels, batch_size):
  assert(len(samples) == len(labels))
  ds = tf.data.Dataset.from_tensor_slices((samples, labels))
  return ds.batch(batch_size)

def load_dataset(dataset_loader,
                 train_size, train_batch_size,
                 test_size, test_batch_size):
  shuffle_seed = current_date()

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset_loader(train_size, 0, test_size)
  train_dataset = create_dataset_from_arrays(x_train, y_train, train_batch_size)
  test_dataset = create_dataset_from_arrays(x_test, y_test, test_batch_size)

  train_dataset = train_dataset.shuffle(len(x_train), shuffle_seed, reshuffle_each_iteration = True)
  return (train_dataset, test_dataset)

def current_date():
  date = datetime.datetime.now()
  return int(date.strftime("%Y%m%d"))

def check_accuracy_greater(accuracy, acc_value):
  logging.getLogger().info("Test accuracy: " % accuracy)
  assert accuracy > acc_value

def compare_weights(weights1, weights2, tolerance):
  wtocompare = list(zip(weights1, weights2))
  for (tensor1, tensor2) in wtocompare:
    assert np.allclose(tensor1, tensor2, atol=tolerance)

