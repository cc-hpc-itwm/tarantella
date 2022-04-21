# CustomLearningRateScheduler:
#   Copyright (C) 2021 keras.io <https://www.tensorflow.org/guide/keras/custom_callback#learning_rate_scheduling>
#   Modifications Copyright (C) 2022 Fraunhofer ITWM <http://www.itwm.fraunhofer.de/>

from models import mnist_models as mnist
import training_runner as base_runner
import callback_utilities
import utilities as util
import tarantella as tnt

import tensorflow as tf
import enum
import pytest

setup_save_path = callback_utilities.setup_save_path

class CallbackWrapper(enum.Enum):
  NONE = 0,
  DEFAULT = 1,  # behaves similar to a normal keras callback - runs only once (on a single rank)
  ONE_RANK = 2,
  MULTI_RANK = 3

class PrintingCallback(tf.keras.callbacks.Callback):
  def on_batch_end(self, batch, logs = None):
    if batch % 4 == 0:
      print(f"[rank {tnt.get_rank()}] Batch {batch} done")

  def on_epoch_end(self, epoch, logs = None):
    print(f"[rank {tnt.get_rank()}] Epoch {epoch} done")

def is_callback_running_on_one_rank(model_config, wrapper_type):
  if wrapper_type in [CallbackWrapper.NONE]:
    if model_config.parallel_strategy == tnt.ParallelStrategy.DATA:
      return False
  elif wrapper_type in [CallbackWrapper.MULTI_RANK]:
    return False
  return True

@pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.subclassed_model_generator),
                                          pytest.param(base_runner.ModelConfig(mnist.fc_model_generator_four_partitions,
                                                                               tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 4,
                                                                                reason="Can only run on 4 ranks, model has 4 partitions")),
                                          pytest.param(base_runner.ModelConfig(mnist.fc_model_generator,
                                                                               tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 1,
                                                                                reason="Cannot run multi-rank, model has only one partition")),
                                          ])
class TestUserDefinedCallbacks:
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("wrapper_type", [CallbackWrapper.NONE,
                                            CallbackWrapper.DEFAULT,
                                            CallbackWrapper.ONE_RANK])
  def test_custom_callback_one_rank(self, model_config, number_epochs, wrapper_type, capsys):
    if not is_callback_running_on_one_rank(model_config, wrapper_type):
      pytest.skip("test configuration for multi-rank callback")

    ref_callback = PrintingCallback()
    tnt_callback = PrintingCallback()
    if wrapper_type == CallbackWrapper.DEFAULT:
      tnt_callback = tnt.keras.callbacks.Callback(tnt_callback)
    elif wrapper_type == CallbackWrapper.ONE_RANK:
      tnt_callback = tnt.keras.callbacks.Callback(tnt_callback, run_on_all_ranks = False)

    train_dataset, _ = callback_utilities.train_val_dataset_generator()
    ref_train_dataset, _ = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, reference_model_runner = callback_utilities.gen_model_runners(model_config)

    param_dict = {'epochs' : number_epochs,
                  'verbose' : 0,
                  'shuffle' : False}
    tnt_history = tnt_model_runner.model.fit(train_dataset,
                                             callbacks = [tnt_callback],
                                             **param_dict)
    tnt_captured = capsys.readouterr()
    ref_history = reference_model_runner.model.fit(ref_train_dataset,
                                                   callbacks = [ref_callback],
                                                   **param_dict)
    ref_captured = capsys.readouterr()

    callback_utilities.assert_identical_tnt_and_ref_history(tnt_history, ref_history)
    if tnt.is_master_rank():
      result = (tnt_captured.out == ref_captured.out)
    else:
      result = all([tnt_captured.out == "", tnt_captured.err == ""])
    util.assert_on_all_ranks(result)

  @pytest.mark.parametrize("number_epochs", [3])
  @pytest.mark.parametrize("wrapper_type", [CallbackWrapper.NONE,
                                            CallbackWrapper.MULTI_RANK])
  def test_custom_callback_multi_rank(self, model_config, number_epochs, wrapper_type, capsys):
    if is_callback_running_on_one_rank(model_config, wrapper_type):
      pytest.skip("test configuration for single-rank callback")

    ref_callback = PrintingCallback()
    tnt_callback = PrintingCallback()
    if wrapper_type == CallbackWrapper.MULTI_RANK:
      tnt_callback = tnt.keras.callbacks.Callback(tnt_callback, run_on_all_ranks = True)

    train_dataset, _ = callback_utilities.train_val_dataset_generator()
    ref_train_dataset, _ = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, reference_model_runner = callback_utilities.gen_model_runners(model_config)

    param_dict = {'epochs' : number_epochs,
                  'verbose' : 0,
                  'shuffle' : False}
    tnt_history = tnt_model_runner.model.fit(train_dataset,
                                             callbacks = [tnt_callback],
                                             **param_dict)
    tnt_captured = capsys.readouterr()
    ref_history = reference_model_runner.model.fit(ref_train_dataset,
                                                   callbacks = [ref_callback],
                                                   **param_dict)
    ref_captured = capsys.readouterr()

    callback_utilities.assert_identical_tnt_and_ref_history(tnt_history, ref_history)
    util.assert_on_all_ranks(tnt_captured.out == ref_captured.out)

