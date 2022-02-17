import tarantella as tnt
import tarantella.keras.callbacks as tnt_callbacks
import tarantella.utilities.tf_version as version_utils

import tensorflow.keras.callbacks as tf_callbacks
import tarantella.strategy.pipelining.pipelining_callbacks
from enum import Enum
from tarantella import logger


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


def _preprocess_callbacks(callbacks, group, parallel_strategy,
                          exec_type = 'fit', verbose = None):
  callbacks = callbacks or []
  _add_default_History_callback_if_necessary(callbacks)
  _add_default_ProgbarLogger_callback_if_necessary(callbacks, exec_type, verbose)
  _to_parallel_callbacks(callbacks, group, parallel_strategy)
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

def _to_parallel_callbacks(callbacks, group, parallel_strategy):
  for index, callback in enumerate(callbacks):
    logger.debug(f"[{parallel_strategy}] Preprocessing callback {callback} of type {type(callback)}")
    callbacks[index] = tnt.keras.callbacks.Callback(callback, parallel_strategy, group = group)
  return callbacks
