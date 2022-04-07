# CustomLearningRateScheduler:
#   Copyright (C) 2021 keras.io <https://www.tensorflow.org/guide/keras/custom_callback#learning_rate_scheduling>
#   Modifications Copyright (C) 2022 Fraunhofer ITWM <http://www.itwm.fraunhofer.de/>

from models import mnist_models as mnist
import training_runner as base_runner
import callback_utilities
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import os
import pytest

setup_save_path = callback_utilities.setup_save_path

class TestModelCheckpointCallback:
  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator)])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_model_checkpoint_data_par(self, setup_save_path, model_config, number_epochs):
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = setup_save_path)]
    callback_utilities.train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs)
    # FIXME: assert correct file exists
    assert True

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator,
                                                                    tnt.ParallelStrategy.ALL)])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_model_checkpoint_pipelining(self, setup_save_path, model_config, number_epochs):
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = setup_save_path)]
    with pytest.raises(ValueError):
      callback_utilities.train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs)

