from tarantella import logger
import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.strategy.pipelining.utilities as putil

import copy
import numpy as np
from functools import singledispatchmethod
from typing import Callable, Dict, Type, Any

import tensorflow as tf

def _on_epoch_end_with_broadcast(self, epoch, logs=None):
  #EarlyStopping, LRonPlateau
  if tnt.is_group_master_rank(self.group):
    monitor_value = self.get_monitor_value(logs, self.monitor)
    self._broadcast_if_necessary(monitor_value)
  else:
    monitor_value = self._broadcast_if_necessary()
    logs[self.monitor] = monitor_value
  return self.keras_callback.on_epoch_end(epoch = epoch, logs = logs)


def _on_batch_end_with_broadcast(self, batch, logs=None):
  # _TerminateOnNaN
  if tnt.is_group_master_rank(self.group):
    loss = self.get_monitor_value(logs, "loss")
    is_loss_nan = (loss is not None) and \
                  (np.isnan(loss) or np.isinf(loss))
    self._broadcast_if_necessary(is_loss_nan)
  else:
    is_loss_nan = self._broadcast_if_necessary()
    # default non-NaN value is needed to detect whether to stop training
    DEFAULT_LOSS = 1.0
    logs["loss"] = np.nan if is_loss_nan else DEFAULT_LOSS
  return self.keras_callback.on_batch_end(batch = batch, logs = logs)


def _generate_pipelining_callback(base_type: Type[tf.keras.callbacks.Callback]) -> Type[tf.keras.callbacks.Callback]:
  class PipeliningCallback(base_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback,
                       aggregate_logs: bool,
                       run_on_all_ranks: bool) -> None:
      logger.debug(f"[PipeliningParallelCallback] Initializing with {keras_callback}")
      super().__init__(keras_callback, aggregate_logs, run_on_all_ranks)
      self._flag_broadcaster = None
      self._tensor_broadcaster = None
      self.customize_callback(keras_callback)
      logger.debug(f"[PipeliningParallelCallback] Configuration: `is_user_defined={self.user_defined_callback}` "
                   f"and `run_on_all_ranks={run_on_all_ranks}`")

    @singledispatchmethod
    def customize_callback(self, keras_callback: tf.keras.callbacks.Callback) -> None:
      logger.debug("[PipeliningParallel] Generic callback")

    @customize_callback.register
    def _(self, keras_callback: tf.keras.callbacks.History):
      logger.debug("[PipeliningParallel] History callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.BaseLogger):
      # Do not support user-added `BaseLogger`s,
      # b/c they do not provide any use
      # and b/c of this issue (https://github.com/tensorflow/tensorflow/issues/46344)
      raise ValueError("[PipeliningParallel] Tarantella does not support "
                        "`tf.keras.callbacks.BaseLogger`")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.CSVLogger):
      logger.debug("[PipeliningParallel] CSVLogger callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.EarlyStopping):
      logger.debug("[PipeliningParallel] EarlyStopping callback")
      self._run_on_all_ranks = True
      # only master rank should print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value
      self.on_epoch_end = lambda *args, **kwargs: _on_epoch_end_with_broadcast(self, *args, **kwargs)
      self._distribute_callback = self._distribute_callback_identity

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.LearningRateScheduler):
      self._run_on_all_ranks = True
      logger.debug("[PipeliningParallel] LearningRateScheduler callback")
      if not tnt.global_tnt_config.output_on_all_devices:
        if not tnt.is_group_master_rank(self.group):
          self.verbose = 0
      self._distribute_callback = self._distribute_callback_identity

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ModelCheckpoint):
      raise ValueError("[PipeliningParallel] Tarantella does not support "
                        "`tf.keras.callbacks.ModelCheckpoint` for partitioned models")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ProgbarLogger):
      logger.debug("[PipeliningParallel] ProgbarLogger callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.ReduceLROnPlateau):
      logger.debug("[PipeliningParallel] ReduceLROnPlateau callback")
      self._run_on_all_ranks = True
      # only master rank should print messages
      self.verbose = keras_callback.verbose if tnt.is_group_master_rank(self.group) \
                                            else utilities.TF_verbose.SILENT.value
      self.on_epoch_end = lambda *args, **kwargs: _on_epoch_end_with_broadcast(self, *args, **kwargs)
      self._distribute_callback = self._distribute_callback_identity

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.RemoteMonitor):
      logger.debug("[PipeliningParallel] RemoteMonitor callback")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TensorBoard):
      logger.debug("[PipeliningParallel] TensorBoard callback")
      utilities._customize_tensorboard_callback(callback = self,
                tensorboard_on_all_devices_env = tnt.global_tnt_config.tensorboard_on_all_devices)

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TerminateOnNaN):
      logger.debug("[PipeliningParallel] TerminateOnNaN callback")
      self._run_on_all_ranks = True
      self.on_batch_end = lambda *args, **kwargs: _on_batch_end_with_broadcast(self, *args, **kwargs)
      self._distribute_callback = self._distribute_callback_identity

    def _process_callback_logs(self, callback_params: Dict) -> Dict:
      kwargs_copy = copy.deepcopy(callback_params)
      if "logs" not in callback_params.keys():
        return kwargs_copy

      user_defined_metrics = putil.extract_user_visible_metrics(kwargs_copy["logs"])
      if len(user_defined_metrics) == 0: # `evaluate` called from `fit`
        return kwargs_copy

      remainder_metrics = putil.remove_user_visible_metrics(kwargs_copy["logs"])
      kwargs_copy["logs"] = putil.avg_metrics_over_pipeline_stages(user_defined_metrics)
      kwargs_copy["logs"].update(remainder_metrics)
      return kwargs_copy

    def _distribute_callback(self, callback_func: Callable, **kwargs: Any) -> Any:
      if self._run_on_all_ranks or \
         tnt.is_group_master_rank(self.group):
        processed_kwargs = self._process_callback_logs(kwargs)
        return callback_func(**processed_kwargs)

    def _distribute_callback_identity(self, callback_func: Callable, **kwargs: Any) -> Any:
      return callback_func(**kwargs)

    def _setup_tensor_broadcaster_if_necessary(self) -> None:
      if self._tensor_broadcaster is None:
         # FIXME: only float values supported
        root_rank_local = self.group.to_group_rank(tnt.get_group_master_rank(self.group))
        self._tensor_broadcaster = tnt.TensorBroadcaster(0., root_rank=root_rank_local,
                                                         group = self.group)

    def _setup_flag_broadcaster_if_necessary(self) -> None:
      if self._flag_broadcaster is None:
        root_rank_local = self.group.to_group_rank(tnt.get_group_master_rank(self.group))
        self._flag_broadcaster = tnt.TensorBroadcaster(True, root_rank=root_rank_local,
                                                       group = self.group)

    def _broadcast_if_necessary(self, value: Any = None) -> Any:
      self._setup_flag_broadcaster_if_necessary()
      if tnt.is_group_master_rank(self.group):
        is_processing_needed = (value is not None)
        self._flag_broadcaster.broadcast(is_processing_needed)
      else:
        is_processing_needed = self._flag_broadcaster.broadcast()

      if is_processing_needed:
        self._setup_tensor_broadcaster_if_necessary()
        broadcasted_value = self._tensor_broadcaster.broadcast(value)
        return broadcasted_value

    def get_monitor_value(self, logs, key):
      logs = logs or {}
      monitor_value = logs.get(key)
      if monitor_value is None:
        logger.warning(f"[PipeliningParallel] Callback conditioned on metric {key} "
                       f"which is not available. Available metrics are: {','.join(list(logs.keys()))}")
      return monitor_value

  return PipeliningCallback
