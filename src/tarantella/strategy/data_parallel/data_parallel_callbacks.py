import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.utilities.tf_version as version_utils
from tarantella import logger

import atexit
import copy
from functools import singledispatchmethod
from typing import Any, Callable, Dict, Iterable, Type

import tensorflow as tf
import numpy as np

class LogsAverager:
  def __init__(self, group: tnt.Group = tnt.Group()) -> None:
    self.group = group
    self.num_ranks = group.size
    self.allreducer = None
    atexit.register(self.close)

  def create_allreducer(self, logs: Dict[str, Any]) -> None:
    self.allreducer = tnt.TensorAllreducer(logs, group = self.group)

  def average_logs(self, logs: Dict[str, Any]) -> Dict[str, Any]:
    if self.allreducer is None:
      self.create_allreducer(logs)
    sum_logs = self.allreducer.allreduce(logs)             # type: ignore [attr-defined]
    average_logs = { k : v / self.num_ranks for k, v in sum_logs.items() }
    return average_logs

  def average_specific_metrics(self, logs: Dict[str, Any],
                                     metric_names: Iterable[str]) -> Dict[str, Any]:
    log_values = dict()
    for k in metric_names:
      if k in logs:
        log_values[k] = logs[k]

    averaged_logs = self.average_logs(log_values)
    for k in averaged_logs:
      logs[k] = averaged_logs[k]
    return logs

  def close(self) -> None:
    del self.allreducer


def _generate_data_parallel_callback(base_type: Type[tf.keras.callbacks.Callback]) -> Type[tf.keras.callbacks.Callback]:
  class DataParallelCallback(LogsAverager, base_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback,
                 aggregate_logs: bool = True,
                 run_on_all_ranks: bool = True,
                 group: tnt.Group = tnt.Group()) -> None:
      super().__init__(group = group)
      logger.debug(f"[DataParallelCallback] init with {keras_callback}")
      base_type.__init__(self, keras_callback)
      self.aggregate_logs = aggregate_logs
      self.run_on_all_ranks = run_on_all_ranks
      self.is_built = False
      self._distribute_callback = self._distribute_callback_default
      self.customize_callback(keras_callback)

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
        averaged_logs = self.average_logs(logs)
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
      # only master rank should save and thus print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value
      self.run_on_all_ranks = False # only one checkpoint is needed (models are identical in a data parallel setting)

      # disable checkpointing for all ranks except the master rank
      if not tnt.is_group_master_rank(self.group):
        self._supports_tf_logs = False
        self.save_freq = 1e20 # very large value to avoid triggering checkpointing
        self.epochs_since_last_save = 0


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
      if tnt.global_tnt_config.tensorboard_on_all_devices:
        self.log_dir += f"/rank_{tnt.get_rank()}"
      else:
        self.run_on_all_ranks = False
        # disable any data logging for all ranks except the master rank
        if not tnt.is_group_master_rank(self.group):
          self.histogram_freq = 0
          self.write_graph = False
          self.write_images = False
          self.write_steps_per_second = False
          self.update_freq = 0
          self.embeddings_freq = 0
          self.embeddings_metadata = None
          self.profile_batch = None

          def _set_model(model):
            pass
          self.set_model = _set_model


    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TerminateOnNaN):
      logger.debug("[DataParallel] TerminateOnNaN callback")

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

    def _average_callback_logs(self, callback_params: Dict[str, Any]) -> Dict[str, Any]:
      kwargs_copy = copy.deepcopy(callback_params)
      # Check if logs do not contain None (for tf versions older than 2.1)
      if kwargs_copy['logs'] is not None:
        if "loss" in kwargs_copy['logs']:
          self._build_tensor_allreducer_if_necessary(kwargs_copy['logs'])
          if self.aggregate_logs:
            kwargs_copy['logs'] = self.average_logs(kwargs_copy['logs'])
      return kwargs_copy

    def _distribute_callback_default(self, callback_func: Callable, **kwargs: Any) -> Any:
      if self.run_on_all_ranks:
        kwargs_copy = self._average_callback_logs(kwargs)    
        return callback_func(**kwargs_copy)
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
    if progbar_logger.run_on_all_ranks:
      kwargs_copy = progbar_logger._average_callback_logs(kwargs)    
      if progbar_logger.should_print_progbar:
        return callback_func(**kwargs_copy)
    else:
      if tnt.is_group_master_rank(progbar_logger.group) and progbar_logger.should_print_progbar:
        return callback_func(**kwargs)
  progbar_logger._distribute_callback = progbar_logger_distribute_callback
