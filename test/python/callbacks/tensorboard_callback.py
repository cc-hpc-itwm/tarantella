from models import mnist_models as mnist
import training_runner as base_runner
import callback_utilities
import utilities as util
import tarantella as tnt

import tensorflow as tf

import enum
import os
import pytest

setup_save_path = callback_utilities.setup_save_path

class CallbackWrapper(enum.Enum):
  NONE = 0,
  DEFAULT = 1,
  ONE_RANK = 2,
  MULTI_RANK = 3

@pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.subclassed_model_generator),
                                          pytest.param(base_runner.ModelConfig(mnist.fc_model_generator_four_partitions,
                                                                               tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 4,
                                                                                reason="Model has 4 partitions, needs 4 ranks to run on")),
                                          ])
class TestTensorboardCallbacks:
  @pytest.mark.parametrize("number_epochs", [1])
  @pytest.mark.parametrize("wrapper_type", [CallbackWrapper.NONE,
                                            CallbackWrapper.DEFAULT,
                                            CallbackWrapper.ONE_RANK])
  def test_tensorboard_one_rank(self, setup_save_path, model_config, number_epochs, wrapper_type):
    (train_dataset, val_dataset) = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, _ = callback_utilities.gen_model_runners(model_config)

    callback = tf.keras.callbacks.TensorBoard(log_dir = setup_save_path)
    if wrapper_type == CallbackWrapper.DEFAULT:
      callback = tnt.keras.callbacks.Callback(callback)
    elif wrapper_type == CallbackWrapper.ONE_RANK:
      callback = tnt.keras.callbacks.Callback(callback, run_on_all_ranks = False)

    tnt_model_runner.model.fit(train_dataset, validation_data=val_dataset,
                               epochs = number_epochs,
                               callbacks = [callback])
    result = [True]
    if tnt.is_master_rank():
      result = [os.path.isdir(os.path.join(setup_save_path, "train")),
                os.path.isdir(os.path.join(setup_save_path, "validation"))]
      result = [all(result)]
    util.assert_on_all_ranks(result)


  @pytest.mark.parametrize("number_epochs", [1])
  @pytest.mark.parametrize("wrapper_type", [CallbackWrapper.MULTI_RANK])
  def test_tensorboard_multi_rank(self, setup_save_path, model_config, number_epochs, wrapper_type):
    (train_dataset, val_dataset) = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, _ = callback_utilities.gen_model_runners(model_config)

    callback = tf.keras.callbacks.TensorBoard(log_dir = setup_save_path)
    if wrapper_type == CallbackWrapper.MULTI_RANK:
      callback = tnt.keras.callbacks.Callback(callback, run_on_all_ranks = True)

    tnt_model_runner.model.fit(train_dataset, validation_data=val_dataset,
                               epochs = number_epochs,
                               callbacks = [callback])
    result = [os.path.isdir(os.path.join(setup_save_path, f"rank_{tnt.get_rank()}/train")),
              os.path.isdir(os.path.join(setup_save_path, f"rank_{tnt.get_rank()}/validation"))]
    result = [all(result)]
    util.assert_on_all_ranks(result)
