import tensorflow as tf
import numpy as np

import tarantella as tnt

class TntModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self, keras_model_checkpoint, underlying_optimizer, distributed_optimizer):
    super(TntModelCheckpoint, self).__init__(keras_model_checkpoint.filepath)
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

def __get_agg_metrics__(input):
  keys, metric_values = np.empty((0)), np.empty((0))
  if input is not None:
    for k in input.keys():
      keys = np.append(keys, k)
      metric_values = np.append(metric_values, input[k])

  allreducer = tnt.TensorAllreducer(metric_values)
  agg_values = allreducer.allreduce(metric_values)

  updated_metrics = dict()
  for i in range(len(keys)):
    updated_metrics[keys[i]] = agg_values[i] / tnt.get_size()

  return updated_metrics

class TntCSVLogger(tf.keras.callbacks.CSVLogger):
  def __init__(self, keras_csvlogger):
    super(TntCSVLogger, self).__init__(keras_csvlogger.filename)

    # set member variables from keras csvlogger instance
    self.sep = keras_csvlogger.sep
    self.append = keras_csvlogger.append

    self.logs = {}
    self.allreducer = None

  def on_train_begin(self, logs=None):
    # only master rank should open a file
    if tnt.is_master_rank():
      super().on_train_begin(logs)

  def on_epoch_end(self, epoch, logs=None):
    if self.allreducer is None:
      self.allreducer = tnt.TensorAllreducer(logs)
    
    # do an allreduce on all ranks and get averaged values over all ranks
    self.logs = self.allreducer.allreduce(logs)
    self.logs.update((k, v / tnt.get_size()) for k, v in self.logs.items())

    # only master rank will write logs to created file
    if tnt.is_master_rank():
      super().on_epoch_end(epoch, self.logs)

  def on_train_end(self, logs=None):
    # only master rank has created a file and thus needs to be closed
    if tnt.is_master_rank():
      super().on_train_end(logs)

class TntHistory(tf.keras.callbacks.History):
  def __init__(self, keras_history):
    super(TntHistory, self).__init__()
    self.logs = {}
    self.allreducer = None

  def on_epoch_end(self, epoch, logs=None):
    if self.allreducer is None:
      self.allreducer = tnt.TensorAllreducer(logs)
    
    # do an allreduce on all ranks and get averaged values over all ranks
    self.logs = self.allreducer.allreduce(logs)
    self.logs.update((k, v / tnt.get_size()) for k, v in self.logs.items())

    super().on_epoch_end(epoch, self.logs)

class TntEarlyStopping(tf.keras.callbacks.EarlyStopping):
  def __init__(self, keras_early_stopping):
    super(TntEarlyStopping, self).__init__()
    self.logs = {}
    self.allreducer = None

    # set member variables from keras csvlogger instance
    self.monitor = keras_early_stopping.monitor
    self.patience = keras_early_stopping.patience
    self.baseline = keras_early_stopping.baseline
    self.min_delta = keras_early_stopping.min_delta
    self.restore_best_weights = keras_early_stopping.restore_best_weights
    self.monitor_op = keras_early_stopping.monitor_op

    # only master rank should save and thus print messages
    self.verbose = keras_early_stopping.verbose if tnt.is_master_rank() else 0

  def get_monitor_value(self, logs):
    if self.allreducer is None:
      self.allreducer = tnt.TensorAllreducer(logs)
    
    # do an allreduce on all ranks and get averaged values over all ranks
    self.logs = self.allreducer.allreduce(logs)
    self.logs.update((k, v / tnt.get_size()) for k, v in self.logs.items())

    super().get_monitor_value(self.logs)
