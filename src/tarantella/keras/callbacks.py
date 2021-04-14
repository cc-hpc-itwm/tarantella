import tensorflow as tf
import numpy as np

import tarantella as tnt

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
  def __init__(self, keras_callback, underlying_optimizer, distributed_optimizer):
    super().__init__(keras_model_checkpoint.filepath)
    self.underlying_optimizer = underlying_optimizer
    self.distributed_optimizer = distributed_optimizer

    # set member variables from ModelCheckpoint instance
    self.validation_data = keras_model_checkpoint.validation_data
    self.model = keras_model_checkpoint.model
    self._chief_worker_only = keras_model_checkpoint._chief_worker_only
    self._supports_tf_logs = True
    self.monitor = keras_model_checkpoint.monitor
    self.filepath = keras_model_checkpoint.filepath
    self.save_best_only = keras_model_checkpoint.save_best_only
    self.save_weights_only = keras_model_checkpoint.save_weights_only
    self.save_freq = keras_model_checkpoint.save_freq
    self.epochs_since_last_save = keras_model_checkpoint.epochs_since_last_save

    if hasattr(keras_model_checkpoint, '_batches_seen_since_last_saving'):  #TF>=2.2
      self._batches_seen_since_last_saving = keras_model_checkpoint._batches_seen_since_last_saving
    if hasattr(keras_model_checkpoint, '_samples_seen_since_last_saving'):  # TF2.0-2.1
      self._samples_seen_since_last_saving = keras_model_checkpoint._samples_seen_since_last_saving

    self._last_batch_seen = 0
    self.load_weights_on_restart = keras_model_checkpoint.load_weights_on_restart
    self.period = keras_model_checkpoint.period
    self.monitor_op = keras_model_checkpoint.monitor_op
    self.best = keras_model_checkpoint.best

    # only master rank should save and thus print messages
    self.verbose = keras_model_checkpoint.verbose if tnt.is_master_rank() else 0

  def on_train_begin(self, logs=None):
    # As of TF 2.3, this only uses `self.model.load_weights`
    super().on_train_begin(logs)

  def on_train_batch_end(self, batch, logs=None):
    # set the optimizer to the underlying to save a plain keras model
    self.model.optimizer = self.underlying_optimizer
    super().on_train_batch_end(batch, logs)
    self.model.optimizer = self.distributed_optimizer

  def on_epoch_end(self, epoch, logs=None):
    # set the optimizer to the underlying to save a plain keras model
    self.model.optimizer = self.underlying_optimizer
    super().on_epoch_end(epoch, logs)
    self.model.optimizer = self.distributed_optimizer



class History(tf.keras.callbacks.History, LogsAverager):
  def __init__(self, keras_callback):
    tf.keras.callbacks.History.__init__(self)
    LogsAverager.__init__(self)

  def on_epoch_end(self, epoch, logs = None):
    averaged_logs = self.average_logs(logs)
    super().on_epoch_end(epoch, averaged_logs)

class EarlyStopping(tf.keras.callbacks.EarlyStopping, LogsAverager):
  def __init__(self, keras_callback):
    tf.keras.callbacks.EarlyStopping.__init__(self)
    LogsAverager.__init__(self)

    # set member variables from keras earlystopping instance
    self.monitor = keras_callback.monitor
    self.patience = keras_callback.patience
    self.baseline = keras_callback.baseline
    self.min_delta = keras_callback.min_delta
    self.restore_best_weights = keras_callback.restore_best_weights
    self.monitor_op = keras_callback.monitor_op

    # only master rank should print messages
    self.verbose = keras_callback.verbose if tnt.is_master_rank() else 0

  def get_monitor_value(self, logs):
    averaged_logs = self.average_logs(logs)
    return super().get_monitor_value(averaged_logs)
