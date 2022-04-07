from tarantella import logger
import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.strategy.pipelining.utilities as putil

import copy
from functools import singledispatchmethod
from typing import Callable, Dict, Type, Any

import tensorflow as tf

def _generate_pipelining_callback(base_type: Type[tf.keras.callbacks.Callback]) -> Type[tf.keras.callbacks.Callback]:
  class PipeliningCallback(base_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback,
                       aggregate_logs: bool,
                       run_on_all_ranks: bool) -> None:
      super().__init__(keras_callback, aggregate_logs, run_on_all_ranks)
      self.customize_callback(keras_callback)

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
    def _(self, keras_callback: tf.keras.callbacks.ModelCheckpoint):
      raise ValueError("[PipeliningParallel] Tarantella does not support "
                        "`tf.keras.callbacks.ModelCheckpoint` for partitioned models")

    @customize_callback.register             # type: ignore [no-redef]
    def _(self, keras_callback: tf.keras.callbacks.TensorBoard):
      logger.debug("[PipeliningParallel] TensorBoard callback")
      utilities._customize_tensorboard_callback(callback = self,
                tensorboard_on_all_devices_env = tnt.global_tnt_config.tensorboard_on_all_devices)

    def _process_callback_logs(self, callback_params: Dict) -> Dict:
      kwargs_copy = copy.deepcopy(callback_params)
      if "logs" not in callback_params.keys():
        return kwargs_copy

      user_defined_metrics = putil.extract_user_visible_metrics(kwargs_copy["logs"])
      if len(user_defined_metrics) == 0: # `evaluate` called from `fit`
        return kwargs_copy

      kwargs_copy["logs"] = putil.avg_metrics_over_pipeline_stages(user_defined_metrics)
      if "loss" in callback_params["logs"]:
        kwargs_copy["logs"]["loss"] = callback_params["logs"]["loss"]
      return kwargs_copy

    def _distribute_callback(self, callback_func: Callable, **kwargs: Any) -> Any:
      if self._run_on_all_ranks or \
         tnt.is_group_master_rank(self.group):
        processed_kwargs = self._process_callback_logs(kwargs)
        return callback_func(**processed_kwargs)

  return PipeliningCallback
