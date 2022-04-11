import tarantella as tnt
import tarantella.utilities.tf_version as version_utils
from tarantella import logger

import tensorflow.keras.callbacks as tf_callbacks

from enum import Enum
import os
import sys

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
  preprocessed_callbacks = _to_parallel_callbacks(callbacks, group, parallel_strategy)
  return preprocessed_callbacks


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
  if progbar_necessary and version_utils.tf_version_above_equal('2.4'):
    # Always need to use `count_mode` to `steps`
    callbacks.append(tf_callbacks.ProgbarLogger(count_mode='steps'))

def _is_progbar_necessary(exec_type, verbose = None):
  if verbose is None:
    progbar_necessary = (TF_DEFAULT_VERBOSE[exec_type] != TF_verbose.SILENT.value)
  else:
    progbar_necessary = (verbose != TF_verbose.SILENT.value)
  return progbar_necessary

def _to_parallel_callbacks(callbacks, group, parallel_strategy):
  parallel_callbacks = []
  for callback in callbacks:
    logger.debug(f"[{parallel_strategy}] Preprocessing callback {callback} of type {type(callback)}")
    parallel_callbacks.append(tnt.keras.callbacks.Callback(callback, parallel_strategy, group = group))
  return parallel_callbacks


def _customize_tensorboard_callback(callback, tensorboard_on_all_devices_env):
  def _tnt_get_log_write_dir(callback):
    return callback.log_dir
  callback._get_log_write_dir = lambda: _tnt_get_log_write_dir(callback)

  def _tnt_set_model(callback, model):
    # redefine `set_model` from here https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L2264
    callback.model = model
    callback._log_write_dir = callback._get_log_write_dir()

    callback._train_dir = os.path.join(callback._log_write_dir, f"train/rank_{tnt.get_rank()}")
    callback._val_dir = os.path.join(callback._log_write_dir, f"validation/rank_{tnt.get_rank()}")

    callback._train_step = callback.model._train_counter  # pylint: disable=protected-access
    callback._val_step = callback.model._test_counter  # pylint: disable=protected-access

    callback._writers = {}  # Resets writers.

    callback._should_write_train_graph = False
    if callback.write_graph:
      callback._write_keras_model_summary()
      callback._should_write_train_graph = True
    if callback.embeddings_freq:
      callback._configure_embeddings()


  if not callback.user_defined_callback:
    # update settings for a TensorBoard callback configured
    # by setting the environment variable TNT_TENSORBOARD_ON_ALL_DEVICES
    callback._run_on_all_ranks = True
  else:
    if callback._run_on_all_ranks != tensorboard_on_all_devices_env:
      logger.warn("[TensorBoard] Conflicting configurations for the callback "
                  f"as `run_on_all_ranks={callback._run_on_all_ranks}` and"
                  f"`TNT_TENSORBOARD_ON_ALL_DEVICES={tensorboard_on_all_devices_env}`. "
                  f"TensorBoard running on {'all ranks' if callback._run_on_all_ranks else 'one rank'}.")

  if (callback.user_defined_callback and callback._run_on_all_ranks) or \
     tensorboard_on_all_devices_env:
    # TF 2.7 (and TF<2.4) hangs when changing the predefined `log_dir` (on train with validation dataset);
    # thus instead of re-naming each rank's `log_dir`, the output will contain
    # multiple files, each corresponding to a rank, but having a random, TF-assigned name
    if version_utils.tf_version_above_equal("2.4") and \
       not version_utils.tf_version_equal("2.7"):
      underlying_keras_callback = callback._get_underlying_callback()
      underlying_keras_callback.set_model = lambda model: _tnt_set_model(underlying_keras_callback, model)

  else:
    callback._run_on_all_ranks = False
    # disregard any data logging for all ranks except the master rank
    if not tnt.is_group_master_rank(callback.group):
      callback._set_underlying_attribute("histogram_freq", 0)
      callback._set_underlying_attribute("write_graph", False)
      callback._set_underlying_attribute("write_images", False)
      callback._set_underlying_attribute("write_steps_per_second", False)
      callback._set_underlying_attribute("update_freq", sys.maxsize)
      callback._set_underlying_attribute("embeddings_freq", 0)
      callback._set_underlying_attribute("embeddings_metadata", None)
      callback._set_underlying_attribute("profile_batch", 0)
  logger.debug(f"[TensorBoard] Callback running on "
               f"{'all ranks' if callback._run_on_all_ranks else 'one rank'}.")

