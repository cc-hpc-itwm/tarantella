
from tarantella import logger
import tarantella as tnt
import tarantella.strategy.pipelining.utilities as putil

import copy
from functools import singledispatchmethod
from typing import Type
import tensorflow as tf

def _generate_pipelining_callback(base_type: Type,
                                  keras_callback: tf.keras.callbacks.Callback,
                                  group: tnt.Group) -> Type:
  class PipeliningCallback(base_type):
    def __init__(self, keras_callback, group = tnt.Group()):
      super().__init__(keras_callback)
      self.group = group
      self.customize_callback(keras_callback)

    @singledispatchmethod
    def customize_callback(self, keras_callback: tf.keras.callbacks.Callback):
      logger.debug("[PipeliningParallel] Generic callback")

    @customize_callback.register
    def _(self, keras_callback: tf.keras.callbacks.History):
      logger.debug("[PipeliningParallel] History callback")

    @customize_callback.register
    def _(self, keras_callback: tf.keras.callbacks.BaseLogger):
      # Do not support user-added `BaseLogger`s,
      # b/c they do not provide any use
      # and b/c of this issue (https://github.com/tensorflow/tensorflow/issues/46344)
      raise ValueError("[PipeliningParallel] Tarantella does not support "
                        "`tf.keras.callbacks.BaseLogger`")

    @customize_callback.register
    def _(self, keras_callback: tf.keras.callbacks.TensorBoard):
      logger.debug("[DataParallel] TensorBoard callback")
      if tnt.global_tnt_config.tensorboard_on_all_devices:
        self.log_dir += f"/rank_{tnt.get_rank()}"
      else:
        # only one rank should actually run the Tensorboard hooks
        def _distribute_callback_tensorboard(callback_func: callable, **kwargs):
          if tnt.is_group_master_rank(self.group):
            return callback_func(**kwargs)
        self._distribute_callback = _distribute_callback_tensorboard

    def _process_callback_logs(self, callback_params: dict):
      if "logs" not in callback_params.keys():
        return callback_params

      kwargs_copy = copy.deepcopy(callback_params)
      user_defined_metrics = putil.extract_user_visible_metrics(kwargs_copy["logs"])
      if len(user_defined_metrics) == 0: # `evaluate` called from `fit`
        return callback_params

      for metric_name, list_of_values in list(user_defined_metrics.items()):
        user_defined_metrics[metric_name] = sum(list_of_values) / len(list_of_values)

      kwargs_copy["logs"] = user_defined_metrics
      if "loss" in callback_params["logs"]:
        kwargs_copy["logs"]["loss"] = callback_params["logs"]["loss"]
      return kwargs_copy

    def _distribute_callback(self, callback_func, **kwargs):
      processed_kwargs = self._process_callback_logs(kwargs)
      if tnt.is_group_master_rank(self.group):
        return callback_func(**processed_kwargs)

  return PipeliningCallback
