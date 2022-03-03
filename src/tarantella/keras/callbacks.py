import tarantella as tnt
import tarantella.strategy.pipelining.pipelining_callbacks as pipecallbacks
import tarantella.strategy.data_parallel.data_parallel_callbacks as dpcallbacks
from tarantella import logger

import copy
from typing import Any, Type
import tensorflow as tf

def _construct_from_keras_object(obj: tf.keras.callbacks.Callback, callback: tf.keras.callbacks.Callback) -> None:
  keras_callback = callback
  if "keras_callback" in keras_callback.__dict__.keys():
    keras_callback = callback.keras_callback
  for k, v in keras_callback.__dict__.items():
    setattr(obj, k, copy.deepcopy(v))


def _generate_default_callback_with_type(tf_callback_type: Type[tf.keras.callbacks.Callback],
                                         parallel_strategy: tnt.ParallelStrategy) -> Type[tf.keras.callbacks.Callback]:

  class DistributedCallback(tf_callback_type):
    def __init__(self, keras_callback: tf.keras.callbacks.Callback) -> None:
      self.keras_callback = keras_callback
      logger.debug(f"Creating generic TNT callback of type={type(keras_callback)}")
      _construct_from_keras_object(self, keras_callback)
      self.tnt_parallel_strategy = parallel_strategy

    def set_params(self, params):
      self.keras_callback.set_params(params)

    def set_model(self, model):
      self.keras_callback.set_model(model)

    def _distribute_callback(self, callback_func, **kwargs):
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
                    callback_type: Type,
                    parallel_strategy: tnt.ParallelStrategy,
                    group: tnt.Group,
                    aggregate_logs: bool = True,
                    run_on_all_ranks: bool = True) -> tf.keras.callbacks.Callback:

  BaseCallback = _generate_default_callback_with_type(callback_type, parallel_strategy)
  DataParallelCallback = dpcallbacks._generate_data_parallel_callback(BaseCallback)
  PipeliningCallback = pipecallbacks._generate_pipelining_callback(BaseCallback)

  if tnt.ParallelStrategy.PIPELINING in parallel_strategy:
    return PipeliningCallback(keras_callback, group)
  else:
    return DataParallelCallback(keras_callback = keras_callback, group = group,
                                aggregate_logs = aggregate_logs, run_on_all_ranks = run_on_all_ranks)


class CallbackMeta(type):
  def __call__(cls, callback: tf.keras.callbacks.Callback,
                    parallel_strategy: tnt.ParallelStrategy = tnt.ParallelStrategy.PIPELINING,
                    group: tnt.Group = tnt.Group(),
                    **kwargs: Any) -> tf.keras.callbacks.Callback:
    if hasattr(callback, "tnt_parallel_strategy"):
      keras_callback_type = type(callback.keras_callback)
    else:
      keras_callback_type = type(callback)
    return callbackFactory(callback, keras_callback_type, parallel_strategy, group, **kwargs)

class Callback(metaclass = CallbackMeta):
  pass
