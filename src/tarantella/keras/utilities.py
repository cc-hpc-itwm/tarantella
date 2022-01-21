import tarantella as tnt
import tarantella.keras.callbacks as tnt_callbacks
import tarantella.utilities.tf_version as version_utils

import tensorflow.keras.callbacks as tf_callbacks
import tarantella.keras.pipelining_callbacks
from enum import Enum


class TF_verbose(Enum):
  SILENT = 0
  ALL = 1
  LESS = 2

# support for TF 2.0 -- 2.3
TF_DEFAULT_VERBOSE = {'fit' : TF_verbose.ALL.value,
                      'evaluate' : TF_verbose.ALL.value,
                      'predict' : TF_verbose.SILENT.value,
                    }

def _set_model_optimizer(model, optimizer):
  if hasattr(model, '_get_optimizer'):
    # wrap optimizer in an internal `keras` data structure
    model.optimizer = model._get_optimizer(optimizer)
  elif hasattr(model, '_set_optimizer'):
    #for Sequential model with TF 2.0/2.1
    model._set_optimizer(optimizer)
  else:
    raise RuntimeError(
    "[tnt.keras.utilities._set_model_optimizer] Cannot set optimizer for the provided `keras.Model`.")


def _preprocess_callbacks(callbacks, group, exec_type = 'fit', verbose = None):
  callbacks = callbacks or []
  _add_default_History_callback_if_necessary(callbacks)
  _add_default_ProgbarLogger_callback_if_necessary(callbacks, exec_type, verbose)
  _to_tnt_callbacks(callbacks, group)
  return callbacks

def _preprocess_pipelining_callbacks(callbacks, group, exec_type = 'fit', verbose = None):
  callbacks = callbacks or []
  _add_default_History_callback_if_necessary(callbacks)
  _add_default_ProgbarLogger_callback_if_necessary(callbacks, exec_type, verbose)

  for index, callback in enumerate(callbacks):
    callbacks[index] = tnt.keras.pipelining_callbacks.callbackFactory(callback)
  return callbacks

def _add_default_History_callback_if_necessary(callbacks):
  for callback in callbacks:
    if isinstance(callback, tf_callbacks.History):
      return
  callbacks.append(tf_callbacks.History())

def _add_default_ProgbarLogger_callback_if_necessary(callbacks, exec_type, verbose):
  for callback in callbacks:
    if isinstance(callback, tf_callbacks.ProgbarLogger):
      return
  progbar_necessary = _is_progbar_necessary(exec_type, verbose)
  if progbar_necessary and version_utils.tf_version_above_equal('2.3'):
    # Always need to use `count_mode` to `steps`
    callbacks.append(tf_callbacks.ProgbarLogger(count_mode='steps'))

def _is_progbar_necessary(exec_type, verbose = None):
  if not verbose:
    progbar_necessary = (TF_DEFAULT_VERBOSE[exec_type] != TF_verbose.SILENT.value)
  else:
    progbar_necessary = (verbose != TF_verbose.SILENT.value)
  return progbar_necessary

def _to_tnt_callbacks(callbacks, group):
  remove_tensorboard_index = None

  for index, callback in enumerate(callbacks):
    # if isinstance(callback, tf_callbacks.ModelCheckpoint):
    #   tnt_callback = tnt_callbacks.ModelCheckpoint(keras_callback = callback,
    #                                                 tnt_model = self)
    #   callbacks[index] = tnt_callback

    if isinstance(callback, tf_callbacks.LearningRateScheduler):
      tnt_callback = tnt_callbacks.LearningRateScheduler(keras_callback = callback)
      callbacks[index] = tnt_callback

    elif isinstance(callback, tf_callbacks.TensorBoard):
      if tnt.global_tnt_config.tensorboard_on_all_devices:
        callback.log_dir += '/rank_{}'.format(tnt.get_rank())
      else:
        if not tnt.is_master_rank():
          remove_tensorboard_index = index

    elif isinstance(callback, tf_callbacks.History):
      hist_callback = tnt_callbacks.History(keras_callback = callback, group = group)
      callbacks[index] = hist_callback

    elif isinstance(callback, tf_callbacks.EarlyStopping):
      early_stopping_callback = tnt_callbacks.EarlyStopping(keras_callback = callback)
      callbacks[index] = early_stopping_callback

    elif isinstance(callback, tf_callbacks.RemoteMonitor):
      remote_monitor_callback = tnt_callbacks.RemoteMonitor(keras_callback = callback)
      callbacks[index] = remote_monitor_callback

    elif isinstance(callback, tf_callbacks.CSVLogger):
      csv_logger_callback = tnt_callbacks.CSVLogger(keras_callback = callback)
      callbacks[index] = csv_logger_callback

    elif isinstance(callback, tf_callbacks.TerminateOnNaN):
      terminate_callback = tnt_callbacks.TerminateOnNaN(keras_callback = callback)
      callbacks[index] = terminate_callback

    elif isinstance(callback, tf_callbacks.BaseLogger):
      # Do not support user-added `BaseLogger`s,
      # b/c they do not provide any use
      # and b/c of this issue (https://github.com/tensorflow/tensorflow/issues/46344)
      raise ValueError("[tnt.Model] Tarantella does not support "
                        "`tf.keras.callbacks.BaseLogger`")

    elif isinstance(callback, tf_callbacks.ReduceLROnPlateau):
      reducelr_callback = tnt_callbacks.ReduceLROnPlateau(keras_callback = callback)
      callbacks[index] = reducelr_callback

    elif isinstance(callback, tf_callbacks.ProgbarLogger):
      progbar_callback = tnt_callbacks.ProgbarLogger(keras_callback = callback, group = group)
      callbacks[index] = progbar_callback
    elif isinstance(callback, tnt_callbacks.Callback):
      callbacks[index] = callback

    elif isinstance(callback, tf_callbacks.Callback):
      custom_callback = tnt_callbacks.Callback(keras_callback=callback)
      callbacks[index] = custom_callback

  if remove_tensorboard_index is not None:
    del callbacks[remove_tensorboard_index]

