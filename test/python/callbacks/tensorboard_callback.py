from models import mnist_models as mnist
import training_runner as base_runner
import callback_utilities
import utilities as util
import tarantella as tnt

import tensorflow as tf

import os
import pytest

setup_save_path = callback_utilities.setup_save_path

@pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator),
                                          base_runner.ModelConfig(mnist.subclassed_model_generator),
                                          pytest.param(base_runner.ModelConfig(mnist.fc_model_generator,
                                                                               tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 1,
                                                                                reason="Cannot run multi-rank, model has only one partition")),
                                          ])
class TestTensorboardCallbacks:
  @pytest.mark.parametrize("number_epochs", [1])
  def test_tensorboard_callback(self, setup_save_path, model_config, number_epochs):
    (train_dataset, val_dataset) = callback_utilities.train_val_dataset_generator()
    tnt_model_runner, _ = callback_utilities.gen_model_runners(model_config)

    tnt_model_runner.model.fit(train_dataset, validation_data=val_dataset,
                               epochs = number_epochs,
                               callbacks = [tf.keras.callbacks.TensorBoard(log_dir = setup_save_path)])
    result = [True]
    if tnt.is_master_rank():
      result = [os.path.isdir(os.path.join(setup_save_path, "train")),
                os.path.isdir(os.path.join(setup_save_path, "validation"))]
      result = [all(result)]
    util.assert_on_all_ranks(result)
