import atexit
import copy
import tensorflow as tf
import numpy as np
import re

import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils



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

def is_real_loss_or_metric(name):
  return "real" in name

def get_element_from_log_name(name, element):
  # name structure: p_{partition_id}_m_{micro_batch_id}_{real/edge/seq}_output_{output_id}_{metric_name}
  # e.g.: `p_1_m_1_real_output_0_sparse_categorical_accuracy`
  assert element in ["micro_batch_id", "", "output_id", "metric_name"]

  m = re.match("p_(?P<partition_id>.+)_m_(?P<micro_batch_id>.+)_(?P<type>.+)_output_(?P<output_id>\d+)_(?P<metric_name>.+)", name)
  return m.groupdict().get(element, None)


def generate_default_callback_with_type(tf_callback_type):
  class Callback(tf_callback_type):
    def __init__(self, keras_callback, *args, **kwargs):
      super().__init__(keras_callback, *args, **kwargs)
#      super().__init__( *args, **kwargs)
      self.keras_callback = keras_callback
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

def callbackFactory(keras_callback, enable_pipelining = True, group = tnt.Group(),
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
      callback_func(**kwargs_copy)
      kwargs_copy = copy.deepcopy(kwargs)
      # Check if logs do not contain None (for tf versions older than 2.1)
      if kwargs_copy['logs'] is not None:
        if "loss" in kwargs_copy['logs']:
          self._build_tensor_allreducer_if_necessary(kwargs_copy['logs'])
          if self.aggregate_logs:
            kwargs_copy['logs'] = self.average_logs(kwargs_copy['logs'])

      if self.run_on_all_ranks:
        callback_func(**kwargs_copy)
      else:
        if tnt.is_group_master_rank(self.group):
          callback_func(**kwargs_copy)


  class PipeliningCallback(BaseCallback):
    def __init__(self, keras_callback, group = tnt.Group()):
      super().__init__(keras_callback)
      self.group = group
      
    # def set_model(self, model):
    #   super().set_model(model)
    #   self.keras_callback.set_model(model)

    def _distribute_callback(self, callback_func, **kwargs):
      if "logs" not in kwargs.keys():
        return kwargs
      
      kwargs_copy = copy.deepcopy(kwargs)
      metrics_per_output = dict()
      for key, value in kwargs["logs"].items():
        if not is_real_loss_or_metric(key):
          continue
        output_id = get_element_from_log_name(key, "output_id")
        metric_name = get_element_from_log_name(key, "metric_name")
        
        if output_id not in metrics_per_output.keys():
          metrics_per_output[output_id] = dict()
        if metric_name not in metrics_per_output[output_id].keys():
          metrics_per_output[output_id][metric_name] = list()
        metrics_per_output[output_id][metric_name].append(value)

      logs = dict()
      for output_id in metrics_per_output.keys():
        for metric_name in metrics_per_output[output_id].keys():
          list_of_values = metrics_per_output[output_id][metric_name]

          new_name = f"output_{output_id}_" if len(metrics_per_output) > 1 else ""
          # loss is unique, not defined per output
          new_name = new_name if metric_name != "loss" else ""

          if metric_name == "loss":
            metric_name = "ploss"
          new_name = new_name + metric_name
          logs[new_name] = sum(list_of_values) / len(list_of_values)
      
      kwargs_copy["logs"] = logs
      if "loss" in kwargs["logs"]:
        kwargs_copy["logs"]["loss"] = kwargs["logs"]["loss"]

      if tnt.is_group_master_rank(self.group):
        callback_func(**kwargs_copy)

  if enable_pipelining:
    return PipeliningCallback(keras_callback, group)
  else:
    return DataParallelCallback(keras_callback, aggregate_logs, run_on_all_ranks, group)
