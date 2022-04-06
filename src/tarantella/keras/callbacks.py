from cProfile import run
import tarantella as tnt
import tarantella.strategy.pipelining.pipelining_callbacks as pipecallbacks
import tarantella.strategy.data_parallel.data_parallel_callbacks as dpcallbacks
from tarantella import logger

import copy
from typing import Any, Type
import tensorflow as tf

def _construct_from_keras_object(obj: tf.keras.callbacks.Callback,
                                 callback: tf.keras.callbacks.Callback) -> None:
  keras_callback = callback
  if "keras_callback" in keras_callback.__dict__.keys():
    keras_callback = callback.keras_callback
  for k, v in keras_callback.__dict__.items():
    setattr(obj, k, copy.deepcopy(v))

def _generate_default_callback_with_type(tf_callback_type: Type[tf.keras.callbacks.Callback],
                                         parallel_strategy: tnt.ParallelStrategy,
                                         group: tnt.Group,
                                         user_defined_callback: bool) -> Type[tf.keras.callbacks.Callback]:

  class DistributedCallback(tf_callback_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback,
                       aggregate_logs: bool,
                       run_on_all_ranks: bool) -> None:
      self.keras_callback = keras_callback
      logger.debug(f"[Callback] Initializing generic tnt.Callback of type={type(keras_callback)}")
      _construct_from_keras_object(self, keras_callback)
      self.tnt_parallel_strategy = parallel_strategy
      self._group = group
      if hasattr(self.keras_callback, "_user_defined_callback"):
        self._user_defined_callback = self.keras_callback._user_defined_callback
        self._aggregate_logs = self.keras_callback._aggregate_logs
        self._run_on_all_ranks = self.keras_callback._run_on_all_ranks

      if hasattr(self, "_user_defined_callback"):
        if user_defined_callback and self._user_defined_callback:
          raise ValueError("[Callback] Cannot wrap a `keras.Callback` twice with user defined settings.")
        self._user_defined_callback = self._user_defined_callback or user_defined_callback

        if user_defined_callback: # enforce user defined parameters
          self._aggregate_logs = aggregate_logs
          self._run_on_all_ranks = run_on_all_ranks
      else:
        self._user_defined_callback = user_defined_callback
        self._aggregate_logs = aggregate_logs
        self._run_on_all_ranks = run_on_all_ranks
      logger.debug(f"[Callback] Configuration: `is_user_defined={self.user_defined_callback}` "
                   f"and `run_on_all_ranks={run_on_all_ranks}`")

    @property
    def group(self):
      return self._group

    @property
    def user_defined_callback(self):
      return self._user_defined_callback

    def set_params(self, params):
      if self._run_on_all_ranks or tnt.is_group_master_rank(self.group):
        self.keras_callback.set_params(params)

    def set_model(self, model):
      if self._run_on_all_ranks or tnt.is_group_master_rank(self.group):
        self.keras_callback.set_model(model)

    def _set_underlying_attribute(self, attribute_name: str,
                                        attribute_value: Any) -> None:
      # ensures that attributes set on nested `tnt.callbacks` go all the way down to the original `keras.callback
      # e.g., `TensorboardCallback` needs to use a modified logging directory for each rank
      setattr(self, attribute_name, attribute_value)
      if hasattr(self.keras_callback, "_set_underlying_attribute"):
        self.keras_callback._set_underlying_attribute(attribute_name, attribute_value)
      else:
        setattr(self.keras_callback, attribute_name, copy.deepcopy(attribute_value))

    def _distribute_callback(self, callback_func, **kwargs):
      if self._run_on_all_ranks or tnt.is_group_master_rank(self.group):
        callback_func(**kwargs)

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
  return DistributedCallback

def callbackFactory(keras_callback: tf.keras.callbacks.Callback,
                    keras_callback_type: Type,
                    parallel_strategy: tnt.ParallelStrategy,
                    group: tnt.Group,
                    aggregate_logs: bool,
                    run_on_all_ranks: bool) -> tf.keras.callbacks.Callback:
  user_defined_callback = (parallel_strategy is None)
  BaseCallback = _generate_default_callback_with_type(keras_callback_type,
                                                      parallel_strategy = parallel_strategy,
                                                      group = group,
                                                      user_defined_callback = user_defined_callback)
  DataParallelCallback = dpcallbacks._generate_data_parallel_callback(BaseCallback)
  PipeliningCallback = pipecallbacks._generate_pipelining_callback(BaseCallback)

  if parallel_strategy == tnt.ParallelStrategy.PIPELINING:
    # default settings for a pipelining callback
    return PipeliningCallback(keras_callback = keras_callback,
                              aggregate_logs = None,
                              run_on_all_ranks = False)
  elif parallel_strategy == tnt.ParallelStrategy.DATA:
    # default settings for a data parallel callback
    return DataParallelCallback(keras_callback = keras_callback,
                                aggregate_logs = True,
                                run_on_all_ranks = True)
  else:
    # unspecified settings lead to the callback behaving like a regular Keras callback,
    # i.e., it is executed on all nodes independently, without any logs aggregation
    aggregate_logs = aggregate_logs if (aggregate_logs is not None) else False
    run_on_all_ranks = run_on_all_ranks if (run_on_all_ranks is not None) else False
    if not run_on_all_ranks and aggregate_logs:
      raise ValueError("[callbackFactory] Cannot aggregate callback logs if callback runs on a single rank")
    return BaseCallback(keras_callback = keras_callback,
                        aggregate_logs = aggregate_logs,
                        run_on_all_ranks = run_on_all_ranks)


class CallbackMeta(type):
  def __call__(cls, callback: tf.keras.callbacks.Callback,
                    parallel_strategy: tnt.ParallelStrategy = None,
                    group: tnt.Group = tnt.Group(),
                    aggregate_logs: bool = None,
                    run_on_all_ranks: bool = None) -> tf.keras.callbacks.Callback:
    if hasattr(callback, "tnt_parallel_strategy"):
      keras_callback_type = type(callback.keras_callback)
    else:
      keras_callback_type = type(callback)
    return callbackFactory(callback,
                           keras_callback_type = keras_callback_type,
                           parallel_strategy = parallel_strategy,
                           group = group,
                           aggregate_logs = aggregate_logs,
                           run_on_all_ranks = run_on_all_ranks)

class Callback(metaclass = CallbackMeta):
  pass
