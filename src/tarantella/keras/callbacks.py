import copy
import tensorflow as tf
import numpy as np

import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils

class LogsAverager(object):
  def __init__(self, num_ranks = tnt.get_size()):
    super().__init__()
    self.num_ranks = num_ranks
    self.allreducer = None

  def average_logs(self, logs):
    if self.allreducer is None:
      self.allreducer = tnt.TensorAllreducer(logs)
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

def _construct_from_keras_object(obj, keras_callback):
  for k, v in keras_callback.__dict__.items():
    setattr(obj, k, copy.deepcopy(v))

class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self, keras_callback, tnt_model):
    super().__init__(filepath = keras_callback.filepath)
    _construct_from_keras_object(self, keras_callback)

    self.tnt_model = tnt_model
    # only master rank should save and thus print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else utilities.TF_verbose.SILENT.value

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
    super().__init__(schedule=keras_callback.schedule, verbose=keras_callback.verbose)
    _construct_from_keras_object(self, keras_callback)

    if not tnt.global_tnt_config.output_on_all_devices:
      if not tnt.is_master_rank():
        self.verbose = 0

class History(LogsAverager, tf.keras.callbacks.History):
  def __init__(self, keras_callback):
    super().__init__()
    _construct_from_keras_object(self, keras_callback)

  def on_epoch_end(self, epoch, logs = None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)

    if version_utils.tf_version_below_equal('2.1'):
      # set the history object, returned by `tnt.Model.fit`,
      # to this callback
      self.model.history = self

class EarlyStopping(LogsAverager, tf.keras.callbacks.EarlyStopping):
  def __init__(self, keras_callback):
    super().__init__()
    _construct_from_keras_object(self, keras_callback)

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else utilities.TF_verbose.SILENT.value

  def get_monitor_value(self, logs):
    averaged_logs = self.average_logs(logs)
    return super().get_monitor_value(averaged_logs)

class RemoteMonitor(LogsAverager, tf.keras.callbacks.RemoteMonitor):
  def __init__(self, keras_callback):
    super().__init__()
    _construct_from_keras_object(self, keras_callback)

  def on_epoch_end(self, epoch, logs=None):
    averaged_logs = self.average_logs(logs)
    if tnt.is_master_rank():
      super().on_epoch_end(epoch, averaged_logs)

class CSVLogger(tf.keras.callbacks.CSVLogger, LogsAverager):
  def __init__(self, keras_callback):
    tf.keras.callbacks.CSVLogger.__init__(self, keras_callback.filename)
    LogsAverager.__init__(self)
    _construct_from_keras_object(self, keras_callback)

  def on_train_begin(self, logs):
    if tnt.is_master_rank():
      super().on_train_begin(logs)

  def on_epoch_end(self, epoch, logs):
    averaged_logs = self.average_logs(logs)
    if tnt.is_master_rank():
      super().on_epoch_end(epoch, averaged_logs)

  def on_train_end(self, logs):
    if tnt.is_master_rank():
      super().on_train_end(logs)

class TerminateOnNaN(LogsAverager, tf.keras.callbacks.TerminateOnNaN):
  def __init__(self, keras_callback):
    super().__init__()
    _construct_from_keras_object(self, keras_callback)

  def on_batch_end(self, batch, logs=None):
    if version_utils.tf_version_below_equal('2.1'):
      averaged_logs = self.average_specific_metrics(logs, self.params['metrics'])
    else:
      averaged_logs = self.average_logs(logs)

    super().on_batch_end(batch, averaged_logs)

class ReduceLROnPlateau(LogsAverager, tf.keras.callbacks.ReduceLROnPlateau):
  def __init__(self, keras_callback):
    super().__init__()
    _construct_from_keras_object(self, keras_callback)

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else utilities.TF_verbose.SILENT.value

  def on_epoch_end(self, epoch, logs=None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)
