import tarantella as tnt
import models.mnist_models as mnist

import tensorflow as tf
import numpy as np

import datetime
import logging
import os
import random

def current_date():
  date = datetime.datetime.now()
  return int(date.strftime("%Y%m%d"))

def create_dataset_from_arrays(samples, labels, batch_size):
  assert(len(samples) == len(labels))
  ds = tf.data.Dataset.from_tensor_slices((samples, labels))
  return ds.batch(batch_size)

def load_dataset(dataset_loader,
                 train_size, train_batch_size,
                 test_size = 0, test_batch_size = 1,
                 shuffle = True):
  set_tf_random_seed()
  shuffle_seed = 1234

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset_loader(train_size, 0, test_size)
  train_dataset = create_dataset_from_arrays(x_train, y_train, train_batch_size)
  test_dataset = create_dataset_from_arrays(x_test, y_test, test_batch_size)

  if shuffle:
    train_dataset = train_dataset.shuffle(len(x_train), shuffle_seed,
                                          reshuffle_each_iteration = False)
  return (train_dataset, test_dataset)

def train_test_mnist_datasets(nbatches = 1, test_nbatches = 0,
                              micro_batch_size = 64, shuffle = True, 
                              extra_batch = 0, extra_sample = 0):
  batch_size = micro_batch_size * tnt.get_size() + extra_batch
  nsamples = nbatches * batch_size + extra_sample
  test_nsamples = test_nbatches * batch_size
  return load_dataset(mnist.load_mnist_dataset,
                      train_size = nsamples, train_batch_size = batch_size,
                      test_size = test_nsamples, test_batch_size = batch_size,
                      shuffle = shuffle)

def set_tf_random_seed(seed = 42):
  np.random.seed(seed)
  tf.random.set_seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  os.environ['TF_DETERMINISTIC_OPS']='1'
  os.environ['TF_CUDNN_DETERMINISTIC']='1'

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
