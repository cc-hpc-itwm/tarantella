import tarantella as tnt
import tarantella.utilities.tf_version as version_utils
import models.mnist_models as mnist

import tensorflow as tf
import numpy as np

import datetime
import logging
import os
import random
import csv
import re

def create_dataset_from_arrays(samples, labels):
  assert(len(samples) == len(labels))
  ds = tf.data.Dataset.from_tensor_slices((samples, labels))
  return ds

def load_dataset(dataset_loader,
                 train_size, train_batch_size,
                 test_size = 0, test_batch_size = 1,
                 shuffle = True, drop_remainder = False):
  set_tf_random_seed()
  shuffle_seed = 1234

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset_loader(train_size, 0, test_size)
  train_dataset = create_dataset_from_arrays(x_train, y_train)
  test_dataset = create_dataset_from_arrays(x_test, y_test)

  if shuffle:
    train_dataset = train_dataset.shuffle(len(x_train), seed = shuffle_seed,
                                          reshuffle_each_iteration = False)
  return train_dataset.batch(train_batch_size,drop_remainder = drop_remainder), \
         test_dataset.batch(test_batch_size,drop_remainder = drop_remainder)

def train_test_mnist_datasets(nbatches = 1, test_nbatches = 0,
                              micro_batch_size = 64, shuffle = True, 
                              remainder_samples_per_batch = 0,
                              last_incomplete_batch_size = 0,
                              drop_remainder = False):
  batch_size = micro_batch_size * tnt.get_size() + remainder_samples_per_batch
  nsamples = nbatches * batch_size + last_incomplete_batch_size
  test_nsamples = test_nbatches * batch_size
  return load_dataset(mnist.load_mnist_dataset,
                      train_size = nsamples, train_batch_size = batch_size,
                      test_size = test_nsamples, test_batch_size = batch_size,
                      shuffle = shuffle, drop_remainder = drop_remainder)

def set_tf_random_seed(seed = 42):
  np.random.seed(seed)
  tf.random.set_seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  os.environ['TF_CUDNN_DETERMINISTIC']='1'

  # from TF 2.7, 'TF_DETERMINISTIC_OPS' was replaced with `enable_op_determinism`
  # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
  if version_utils.tf_version_below_equal('2.6'):
    os.environ['TF_DETERMINISTIC_OPS']='1'


def same_random_int_all_ranks(low, high):
  set_tf_random_seed()
  return random.randint(low, high)

def check_accuracy_greater(accuracy, acc_value):
  logging.getLogger().info("Test accuracy: {}".format(accuracy))
  assert accuracy > acc_value

def compare_weights(weights1, weights2, tolerance):
  wtocompare = list(zip(weights1, weights2))
  for (tensor1, tensor2) in wtocompare:
    assert np.allclose(tensor1, tensor2, atol=tolerance)

def check_model_configuration_identical(model1, model2):
  config1 = model1.get_config()['layers']
  config2 = model2.get_config()['layers']
  assert config1 == config2

def tuple_to_list_in_dictionary(dictionary):
  for key, value in dictionary.items():
    if isinstance(value, tuple):
      dictionary[key] = list(value)
  return dictionary

def update_configuration(model_config):
  for layer in model_config:
    if 'config' in layer:
      layer['config'] = tuple_to_list_in_dictionary(layer['config'])
  return model_config

def check_model_configuration_identical_legacy(model1, model2):
  # for TF2.1/2.0, comparing configurations directly fails because
  # model load converts tuple values to lists (e.g., tensor shapes)
  config1 = model1.get_config()['layers']
  config2 = model2.get_config()['layers']

  config1 = update_configuration(config1)
  config2 = update_configuration(config2)
  assert config1 == config2

def get_metric_values_from_file(filename):
  metrics = []
  with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
      metrics += [float(value) for value in row.values()]
  return metrics

def get_metrics_from_stdout(captured_text, metric_names):
  metrics = []
  for name in metric_names:
    search_string = " " + name + ": (\d+(?:\.\d+)?)"

    # Returns an array of values for particular metric
    metrics += re.findall(search_string, captured_text, re.IGNORECASE)

  return [float(m) for m in metrics]
