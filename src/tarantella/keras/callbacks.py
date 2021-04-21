import copy
import tensorflow as tf
import numpy as np

import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils

class LogsAverager():
  def __init__(self, num_ranks = tnt.get_size()):
    self.num_ranks = num_ranks
    self.allreducer = None

  def average_logs(self, logs):
    if self.allreducer is None:
      self.allreducer = tnt.TensorAllreducer(logs)
    sum_logs = self.allreducer.allreduce(logs)
    average_logs = { k : v / self.num_ranks for k, v in sum_logs.items() }
    return average_logs

class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self, keras_callback, distributed_optimizer):
    self._construct_from_keras_object(keras_callback)
    self.distributed_optimizer = distributed_optimizer
    # only master rank should save and thus print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else 0

  def _construct_from_keras_object(self, keras_callback):
    implemented_methods = ['on_epoch_end',
                           'on_train_begin',
                           'on_train_batch_end' ]
    super().__init__(keras_callback.filepath)
    for k, v in keras_callback.__dict__.items():
      if k not in implemented_methods:
        setattr(self, k, copy.deepcopy(v))

  def on_train_begin(self, logs=None):
    # As of TF 2.3, this only uses `self.model.load_weights`
    super().on_train_begin(logs)

  def on_train_batch_end(self, batch, logs=None):
    # set the optimizer to the underlying to save a plain keras model
    utilities._set_model_optimizer(self.model, self.distributed_optimizer.underlying_optimizer)
    super().on_train_batch_end(batch, logs)
    utilities._set_model_optimizer(self.model, self.distributed_optimizer)

  def on_epoch_end(self, epoch, logs=None):
    # set the optimizer to the underlying to save a plain keras model
    utilities._set_model_optimizer(self.model, self.distributed_optimizer.underlying_optimizer)
    super().on_epoch_end(epoch, logs)
    utilities._set_model_optimizer(self.model, self.distributed_optimizer)

class LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
  def __init__(self, keras_callback):
    super().__init__(schedule=keras_callback.schedule, verbose=keras_callback.verbose)

    if not tnt.global_tnt_config.output_on_all_devices:
      if not tnt.is_master_rank():
        self.verbose = 0


class History(tf.keras.callbacks.History, LogsAverager):
  def __init__(self, keras_callback):
    tf.keras.callbacks.History.__init__(self)
    LogsAverager.__init__(self)

  def on_epoch_end(self, epoch, logs = None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)

    if version_utils.tf_version_below_equal('2.1'):
      # set the history object, returned by `tnt.Model.fit`,
      # to this callback
      self.model.history = self

class EarlyStopping(tf.keras.callbacks.EarlyStopping, LogsAverager):
  def __init__(self, keras_callback):
    self._construct_from_keras_object(keras_callback)
    LogsAverager.__init__(self)

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else 0

  def _construct_from_keras_object(self, keras_callback):
    implemented_methods = ['get_monitor_value']
    super().__init__()
    for k, v in keras_callback.__dict__.items():
      if k not in implemented_methods:
        setattr(self, k, copy.deepcopy(v))

  def get_monitor_value(self, logs):
    averaged_logs = self.average_logs(logs)
    return super().get_monitor_value(averaged_logs)
