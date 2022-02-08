import atexit
import copy
import tensorflow as tf
import numpy as np
import re

from tarantella import logger
import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils
import tarantella.strategy.pipelining.utilities as putil


class LogsAverager(object):
  def __init__(self, group = tnt.Group()):
    super().__init__()
    self.group = group
    self.num_ranks = group.size
    self.allreducer = None
    atexit.register(self.close)

  def create_allreducer(self, logs):
    self.allreducer = tnt.TensorAllreducer(logs, group = self.group)

  def average_logs(self, logs):
    if self.allreducer is None:
      self.create_allreducer(logs)
    sum_logs = self.allreducer.allreduce(logs)
    average_logs = { k : v / self.num_ranks for k, v in sum_logs.items() }
    return average_logs

  def average_specific_metrics(self, logs, metric_names):
    log_values = dict()
    for k in metric_names:
      if k in logs:
        log_values[k] = logs[k]

    averaged_logs = self.average_logs(log_values)
    for k in averaged_logs:
      logs[k] = averaged_logs[k]

    return logs

  def close(self):
    del self.allreducer

def _construct_from_keras_object(obj, keras_callback):
  for k, v in keras_callback.__dict__.items():
    if k not in ["group"]:
      setattr(obj, k, copy.deepcopy(v))

def generate_default_callback_with_type(tf_callback_type):
  class Callback(tf_callback_type):
    def __init__(self, keras_callback):
      self.keras_callback = keras_callback
      logger.debug(f"Creating generic TNT callback of type={type(keras_callback)}")
      _construct_from_keras_object(self, keras_callback)
    
    def set_params(self, params):
      self.keras_callback.set_params(params)
    
    def set_model(self, model):
      self.keras_callback.set_model(model)

    def _distribute_callback(self, callback_func, **kwargs):
      callback_func(**kwargs_copy)
    
    def on_epoch_begin(self, epoch, logs=None):
      self._distribute_callback(self.keras_callback.on_epoch_begin, epoch=epoch, logs=logs)
    
    def on_epoch_end(self, epoch, logs=None):
      self._distribute_callback(self.keras_callback.on_epoch_end, epoch=epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_batch_begin, batch=batch, logs=logs)
    
    def on_batch_end(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_batch_end, batch=batch, logs=logs)

    def on_train_batch_begin(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_train_batch_begin, batch=batch, logs=logs)
    
    def on_train_batch_end(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_train_batch_end, batch=batch, logs=logs)
    
    def on_test_batch_begin(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_test_batch_begin, batch=batch, logs=logs)
    
    def on_test_batch_end(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_test_batch_end, batch=batch, logs=logs)
    
    def on_predict_batch_begin(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_predict_batch_begin, batch=batch, logs=logs)
    
    def on_predict_batch_end(self, batch, logs=None):
      self._distribute_callback(self.keras_callback.on_predict_batch_end, batch=batch, logs=logs)
    
    def on_train_begin(self, logs=None):
      self._distribute_callback(self.keras_callback.on_train_begin, logs=logs)
    
    def on_train_end(self, logs=None):
      self._distribute_callback(self.keras_callback.on_train_end, logs=logs)
    
    def on_test_begin(self, logs=None):
      self._distribute_callback(self.keras_callback.on_test_begin, logs=logs)
    
    def on_test_end(self, logs=None):
      self._distribute_callback(self.keras_callback.on_test_end, logs=logs)
    
    def on_predict_begin(self, logs=None):
      self._distribute_callback(self.keras_callback.on_predict_begin, logs=logs)
    
    def on_predict_end(self, logs=None):
      self._distribute_callback(self.keras_callback.on_predict_end, logs=logs)
    
    def _implements_train_batch_hooks(self):
      return self.keras_callback._implements_train_batch_hooks()

    def _implements_test_batch_hooks(self):
      return self.keras_callback._implements_test_batch_hooks()
    
    def _implements_predict_batch_hooks(self):
      return self.keras_callback._implements_predict_batch_hooks()
  return Callback

def callbackFactory(keras_callback,
                    parallel_strategy = tnt.ParallelStrategy.PIPELINING, group = tnt.Group(),
                    aggregate_logs=True, run_on_all_ranks=True):

  BaseCallback = generate_default_callback_with_type(type(keras_callback))

  class DataParallelCallback(LogsAverager, BaseCallback):
    def __init__(self, keras_callback,
                 aggregate_logs=True, run_on_all_ranks=True,
                 group = tnt.Group()):
      super().__init__(group = group)
      BaseCallback.__init__(keras_callback)
      self.aggregate_logs = aggregate_logs
      self.run_on_all_ranks = run_on_all_ranks
      self.is_built = False

    def _logs_as_tensors(self, logs):
      logs_as_tensors = copy.deepcopy(logs)
      for key in logs_as_tensors.keys():
        if not tf.is_tensor(logs_as_tensors[key]):
          logs_as_tensors[key] = tf.constant(logs_as_tensors[key])
      return logs_as_tensors

    def _setup_tensor_allreducer(self, logs):
      full_logs = copy.deepcopy(logs)
      for key in list(logs):
        new_key = 'val_' + key
        full_logs[new_key] = np.double(0)
      self.create_allreducer(self._logs_as_tensors(full_logs))

    def _build_tensor_allreducer_if_necessary(self, logs):
      if not self.is_built:
        self._setup_tensor_allreducer(logs)
        self.is_built = True

    def _distribute_callback(self, callback_func, **kwargs):
      kwargs_copy = copy.deepcopy(kwargs)
      # Check if logs do not contain None (for tf versions older than 2.1)
      if kwargs_copy['logs'] is not None:
        if "loss" in kwargs_copy['logs']:
          self._build_tensor_allreducer_if_necessary(kwargs_copy['logs'])
          if self.aggregate_logs:
            kwargs_copy['logs'] = self.average_logs(kwargs_copy['logs'])

      if self.run_on_all_ranks:
        return callback_func(**kwargs_copy)
      else:
        if tnt.is_group_master_rank(self.group):
          return callback_func(**kwargs_copy)


  class PipeliningCallback(BaseCallback):
    def __init__(self, keras_callback, group = tnt.Group()):
      super().__init__(keras_callback)
      self.group = group

    def _distribute_callback(self, callback_func, **kwargs):
      if "logs" not in kwargs.keys():
        return kwargs
      kwargs_copy = copy.deepcopy(kwargs)
      user_defined_metrics = putil.extract_user_visible_metrics(kwargs_copy["logs"])
      for metric_name, list_of_values in list(user_defined_metrics.items()):
        user_defined_metrics[metric_name] = sum(list_of_values) / len(list_of_values)
      
      kwargs_copy["logs"] = user_defined_metrics
      if "loss" in kwargs["logs"]:
        kwargs_copy["logs"]["loss"] = kwargs["logs"]["loss"]

      if tnt.is_group_master_rank(self.group):
        return callback_func(**kwargs_copy)

  if tnt.ParallelStrategy.PIPELINING in parallel_strategy:
    return PipeliningCallback(keras_callback, group)
  else:
    return DataParallelCallback(keras_callback, aggregate_logs, run_on_all_ranks, group)
