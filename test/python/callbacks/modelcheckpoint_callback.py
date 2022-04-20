from models import mnist_models as mnist
import callback_utilities
import training_runner as base_runner
import utilities as util

import tarantella as tnt

import tensorflow as tf
import os
import pytest

setup_save_path = callback_utilities.setup_save_path

class TestModelCheckpointCallback:
  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator)])
  def test_model_checkpoint_data_par(self, setup_save_path, model_config):
    train_dataset, val_dataset = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, _ = callback_utilities.gen_model_runners(model_config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = setup_save_path)]
    tnt_model_runner.model.fit(train_dataset, validation_data=val_dataset, epochs = 1,
                               callbacks = callbacks)
    # only the master rank should save the model
    # the other ranks may use the same shared directory or a local path as their `filepath`
    result = [True]
    if tnt.is_master_rank():
      result = [os.path.isdir(os.path.join(setup_save_path, "assets")),
                os.path.isdir(os.path.join(setup_save_path, "variables"))]
      result = [all(result)]
    util.assert_on_all_ranks(result)

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator,
                                                                    tnt.ParallelStrategy.ALL)])
  def test_model_checkpoint_pipelining(self, setup_save_path, model_config):
    train_dataset, _ = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, _ = callback_utilities.gen_model_runners(model_config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = setup_save_path)]
    with pytest.raises(ValueError):
      tnt_model_runner.model.fit(train_dataset, epochs = 1,
                                 callbacks = callbacks)

