import atexit
import copy
import tensorflow as tf
import numpy as np

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

class Callback(LogsAverager, tf.keras.callbacks.Callback):
  def __init__(self, keras_callback,
               aggregate_logs=True, run_on_all_ranks=True,
               group = tnt.Group()):
    super().__init__(group = group)
    self.aggregate_logs = aggregate_logs
    self.run_on_all_ranks = run_on_all_ranks
    self.keras_callback = keras_callback
    self.is_built = False
    _construct_from_keras_object(self, keras_callback)
  
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

  def set_params(self, params):
    self.keras_callback.set_params(params)
  
  def set_model(self, model):
    self.keras_callback.set_model(model)

  def _distribute_callback(self, callback_func, **kwargs):
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


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self, keras_callback, tnt_model, group = tnt.Group()):
    super().__init__(filepath = keras_callback.filepath)
    _construct_from_keras_object(self, keras_callback)

    self.tnt_model = tnt_model
    # only master rank should save and thus print messages
    self.verbose = keras_callback.verbose if tnt.is_group_master_rank(group) else utilities.TF_verbose.SILENT.value

  def set_model(self, model):
    # Overriding this method ensures that `ModelCheckpoint` is called on the
    # `tnt.Model` within `fit` instead of the internal model
    self.model = self.tnt_model

  def on_train_begin(self, logs=None):
    # As of TF 2.3, this only uses `self.model.load_weights`
    super().on_train_begin(logs)

  def on_train_batch_end(self, batch, logs=None):
    super().on_train_batch_end(batch, logs)

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch, logs)

class LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
  def __init__(self, keras_callback):
    super().__init__(schedule=keras_callback.schedule, verbose=keras_callback.verbose, group = tnt.Group())
    _construct_from_keras_object(self, keras_callback)

    if not tnt.global_tnt_config.output_on_all_devices:
      if not tnt.is_group_master_rank(group):
        self.verbose = 0

class History(LogsAverager, tf.keras.callbacks.History):
  def __init__(self, keras_callback, group = tnt.Group()):
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)

  def on_epoch_end(self, epoch, logs = None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)

    if version_utils.tf_version_below_equal('2.1'):
      # set the history object, returned by `tnt.Model.fit`,
      # to this callback
      self.model.history = self

class EarlyStopping(LogsAverager, tf.keras.callbacks.EarlyStopping):
  def __init__(self, keras_callback, group = tnt.Group()):
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_group_master_rank(group) else utilities.TF_verbose.SILENT.value

  def get_monitor_value(self, logs):
    averaged_logs = self.average_logs(logs)
    return super().get_monitor_value(averaged_logs)

class RemoteMonitor(LogsAverager, tf.keras.callbacks.RemoteMonitor):
  def __init__(self, keras_callback, group = tnt.Group()):
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)

  def on_epoch_end(self, epoch, logs=None):
    averaged_logs = self.average_logs(logs)
    if tnt.is_group_master_rank(self.group):
      super().on_epoch_end(epoch, averaged_logs)

class CSVLogger(tf.keras.callbacks.CSVLogger, LogsAverager):
  def __init__(self, keras_callback, group = tnt.Group()):
    tf.keras.callbacks.CSVLogger.__init__(self, keras_callback.filename)
    LogsAverager.__init__(self, group = group)
    _construct_from_keras_object(self, keras_callback)

  def on_train_begin(self, logs):
    if tnt.is_group_master_rank(self.group):
      super().on_train_begin(logs)

  def on_epoch_end(self, epoch, logs):
    averaged_logs = self.average_logs(logs)
    if tnt.is_group_master_rank(self.group):
      super().on_epoch_end(epoch, averaged_logs)

  def on_train_end(self, logs):
    if tnt.is_group_master_rank(self.group):
      super().on_train_end(logs)

class TerminateOnNaN(LogsAverager, tf.keras.callbacks.TerminateOnNaN):
  def __init__(self, keras_callback, group = tnt.Group()):
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)

  def on_batch_end(self, batch, logs=None):
    if version_utils.tf_version_below_equal('2.1'):
      averaged_logs = self.average_specific_metrics(logs, self.params['metrics'])
    else:
      averaged_logs = self.average_logs(logs)

    super().on_batch_end(batch, averaged_logs)

class ReduceLROnPlateau(LogsAverager, tf.keras.callbacks.ReduceLROnPlateau):
  def __init__(self, keras_callback, group = tnt.Group()):
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_group_master_rank(group) else utilities.TF_verbose.SILENT.value

  def on_epoch_end(self, epoch, logs=None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)

class ProgbarLogger(LogsAverager, tf.keras.callbacks.ProgbarLogger):
  def __init__(self, keras_callback, group = tnt.Group()):
    if version_utils.tf_version_below_equal('2.2'):
      raise EnvironmentError("[tnt.callbacks.ProgbarLogger] "
                             "`ProgbarLogger` support from TF 2.3")
    super().__init__(group = group)
    _construct_from_keras_object(self, keras_callback)
    self.is_built = False
    self.should_print_progbar = tnt.is_group_master_rank(group) # the other ranks only need to participate in averaging logs

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

  def on_epoch_begin(self, epoch, logs=None):
    if self.should_print_progbar:
      super().on_epoch_begin(epoch, logs)

  def on_train_begin(self, logs=None):
    self._called_in_fit = True
    if self.should_print_progbar:
      super().on_train_begin(logs)

  def on_test_begin(self, logs=None):
    if self.should_print_progbar:
      super().on_test_begin(logs)

  def on_predict_begin(self, logs=None):
    if self.should_print_progbar:
      super().on_predict_begin(logs)

  def on_train_batch_end(self, batch, logs=None):
    self._build_tensor_allreducer_if_necessary(logs)
    averaged_logs = self.average_specific_metrics(logs, list(logs.keys()))
    if self.should_print_progbar:
      super().on_train_batch_end(batch, averaged_logs)

  def on_test_batch_end(self, batch, logs=None):
    # FIXME: Average in case validate/evaluate is distributed
    if not self._called_in_fit:
      if self.should_print_progbar:
        super().on_test_batch_end(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    # FIXME: Average in case predict is distributed
    if self.should_print_progbar:
      super().on_predict_batch_end(batch, logs)

  def on_epoch_end(self, epoch, logs=None):
    self._build_tensor_allreducer_if_necessary(logs)
    averaged_logs = self.average_logs(logs)
    if self.should_print_progbar:
      super().on_epoch_end(epoch, averaged_logs)

  def on_test_end(self, logs=None):
    # FIXME: Average in case validate/evaluate is distributed
    if not self._called_in_fit:
      if self.should_print_progbar:
        super().on_test_end(logs)

  def on_predict_end(self, logs=None):
    # FIXME: Average in case predict is distributed
    if self.should_print_progbar:
      super().on_predict_end(logs = None)
