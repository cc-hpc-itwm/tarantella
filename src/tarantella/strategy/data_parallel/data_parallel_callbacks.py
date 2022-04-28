import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils
from tarantella import logger

import atexit
import copy
from functools import singledispatchmethod
from typing import Any, Callable, Dict, List, Type

import tensorflow as tf
import sys

def _get_train_metrics_from_logs(logs: Dict[str, Any]) -> Dict[str, Any]:
  train_metrics = {key: val for key, val in logs.items() if not key.startswith('val_')}
  return train_metrics

def _get_val_metrics_from_logs(logs: Dict[str, Any]) -> Dict[str, Any]:
  val_metrics = {key: val for key, val in logs.items() if key.startswith('val_')}
  return val_metrics

class LogsAverager:
  def __init__(self, group: tnt.Group = tnt.Group()) -> None:
    self.group = group
    self.num_ranks = group.size
    self.initial_metrics = list()
    self.metrics_allreducer = None
    self.val_metrics_allreducer = None
    atexit.register(self._close)

  def average_logs(self, logs: Dict[str, Any]) -> Dict[str, Any]:
    if self.metrics_allreducer is None:
      self._setup_tensor_allreducers(logs)

    # remove additional metrics that did not belong to the initial list used to set up the Allreducer's
    clean_logs = {k : v for k,v in logs.items() if k in self.initial_metrics}

    train_metrics = _get_train_metrics_from_logs(clean_logs)
    sum_logs = self.metrics_allreducer.allreduce(train_metrics)         # type: ignore [attr-defined]

    val_metrics = _get_val_metrics_from_logs(clean_logs)
    if len(val_metrics) > 0:
      sum_val_logs = self.val_metrics_allreducer.allreduce(val_metrics) # type: ignore [attr-defined]
      sum_logs.update(sum_val_logs)

    average_logs = { k : v / self.num_ranks for k, v in sum_logs.items() }
    return average_logs

  def _setup_tensor_allreducers(self, logs: Dict[str, Any]) -> None:
    # separate metrics for training and validation by key
    # use the same initial values for both types of metrics (they should have the same value data types)
    train_metrics = _get_train_metrics_from_logs(logs)
    val_metrics = {f"val_{key}": val for key, val in train_metrics.items()}
    self.initial_metrics = list(train_metrics.keys()) + list(val_metrics.keys())

    self.metrics_allreducer = tnt.TensorAllreducer(train_metrics, group = self.group)
    self.val_metrics_allreducer = tnt.TensorAllreducer(val_metrics, group = self.group)

    # one initial allreduce operation is necessary to set up TensorAllreducer's for each value in the logs
    self.metrics_allreducer.allreduce(train_metrics)
    self.val_metrics_allreducer.allreduce(val_metrics)

  def _close(self) -> None:
    del self.metrics_allreducer
    del self.val_metrics_allreducer


def _generate_data_parallel_callback(base_type: Type[tf.keras.callbacks.Callback]) -> Type[tf.keras.callbacks.Callback]:
  class DataParallelCallback(base_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback,
                       aggregate_logs: bool,
                       run_on_all_ranks: bool) -> None:
      logger.debug(f"[DataParallelCallback] Initializing with {keras_callback}")
      super().__init__(keras_callback, aggregate_logs, run_on_all_ranks)
      self._logs_averager = LogsAverager(self.group)
      self._is_built = False
      self._distribute_callback = self._distribute_callback_default
      self.customize_callback(keras_callback)
      logger.debug(f"[DataParallelCallback] Configuration: `is_user_defined={self.user_defined_callback}` "
                   f"and `run_on_all_ranks={self._run_on_all_ranks}`")

    @singledispatchmethod
    def customize_callback(self, keras_callback: tf.keras.callbacks.Callback) -> None:
      logger.debug("[DataParallel] Generic callback")

    @customize_callback.register
    def _(self, keras_callback: tf.keras.callbacks.BaseLogger):
      # Do not support user-added `BaseLogger`s,
      # b/c they do not provide any use
      # and b/c of this issue (https://github.com/tensorflow/tensorflow/issues/46344)
      raise ValueError("[DataParallel] Tarantella does not support "
                        "`tf.keras.callbacks.BaseLogger`")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.CSVLogger):
      logger.debug("[DataParallel] CSVLogger callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.EarlyStopping):
      logger.debug("[DataParallel] EarlyStopping callback")
      # only master rank should print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value

      def _get_monitor_value(self, logs):
        averaged_logs = self._logs_averager.average_logs(logs)
        return super().get_monitor_value(averaged_logs)
      self.get_monitor_value = _get_monitor_value

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.History):
      logger.debug("[DataParallel] History callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.LearningRateScheduler):
      logger.debug("[DataParallel] LearningRateScheduler callback")
      if not tnt.global_tnt_config.output_on_all_devices:
        if not tnt.is_group_master_rank(self.group):
          self.verbose = 0

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ModelCheckpoint):
      logger.debug("[DataParallel] ModelCheckpoint callback")
      self._run_on_all_ranks = False
      self._aggregate_logs = False
      # only master rank should save and thus print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value
      self._chief_worker_only = True
      # only one checkpoint is needed (models are identical in a data parallel setting)
      if not tnt.is_group_master_rank(self.group):
        self._supports_tf_logs = False
        self.save_freq = sys.maxsize # very large value to avoid triggering checkpointing
        self.epochs_since_last_save = 0
        self.period = sys.maxsize

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ProgbarLogger):
      logger.debug("[DataParallel] ProgbarLogger callback")
      _customize_progbar_logger(self)

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ReduceLROnPlateau):
      logger.debug("[DataParallel] ReduceLROnPlateau callback")
      # only master rank should print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.RemoteMonitor):
      logger.debug("[DataParallel] RemoteMonitor callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TensorBoard):
      logger.debug("[DataParallel] TensorBoard callback")
      self._aggregate_logs = None
      utilities._customize_tensorboard_callback(callback = self,
                tensorboard_on_all_devices_env = tnt.global_tnt_config.tensorboard_on_all_devices)

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TerminateOnNaN):
      logger.debug("[DataParallel] TerminateOnNaN callback")

    def _build_tensor_allreducer_if_necessary(self, logs):
      if not self._is_built:
        self._logs_averager._setup_tensor_allreducers(logs)
        self._is_built = True

    def _average_callback_logs(self, logs: Dict[str, Any]) -> Dict[str, Any]:
      logs_copy = copy.deepcopy(logs)
      # Check if logs do not contain None (for tf versions older than 2.1)
      if logs_copy is not None:
        if "loss" in logs_copy:
          if self._aggregate_logs:
            self._build_tensor_allreducer_if_necessary(logs_copy)
            logs_copy = self._logs_averager.average_logs(logs_copy)
      return logs_copy

    def _distribute_callback_default(self, callback_func: Callable, **kwargs: Any) -> Any:
      if self._run_on_all_ranks:
        averaged_logs = self._average_callback_logs(kwargs['logs'])
        kwargs['logs'].update(averaged_logs)
        return callback_func(**kwargs)
      else:
        if tnt.is_group_master_rank(self.group):
          return callback_func(**kwargs)

  return DataParallelCallback

def _customize_progbar_logger(progbar_logger: tf.keras.callbacks.ProgbarLogger) -> None:
  if version_utils.tf_version_below_equal('2.2'):
    raise EnvironmentError("[tnt.callbacks.ProgbarLogger] "
                            "`ProgbarLogger` support from TF 2.3")
  # the other ranks only need to participate in averaging logs
  progbar_logger.should_print_progbar = tnt.is_group_master_rank(progbar_logger.group)

  def progbar_logger_distribute_callback(callback_func: Callable,
                                         **kwargs: Any) -> Any:
    if progbar_logger._run_on_all_ranks:
      averaged_logs = progbar_logger._average_callback_logs(kwargs['logs'])
      kwargs['logs'].update(averaged_logs)
      if progbar_logger.should_print_progbar:
        return callback_func(**kwargs)
    else:
      if progbar_logger.should_print_progbar:
        return callback_func(**kwargs)
  progbar_logger._distribute_callback = progbar_logger_distribute_callback
