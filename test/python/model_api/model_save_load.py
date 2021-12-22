from models import mnist_models as mnist
import utilities as util

import tarantella as tnt
import tensorflow as tf
from tensorflow import keras

import copy
import numpy as np
import os
import pytest

import shutil
import tempfile

# Saving/loading the whole model applies to keras models and Sequential model.
# Subclassed models are a special case of keras.Model, which is not impacted by
# tarantella saving/loading, as it is handled by `tf.keras`
# (https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading)
@pytest.fixture(scope="class", params=[mnist.fc_model_generator,
                                       mnist.sequential_model_generator])
def model(request):
  yield request.param()

@pytest.fixture(scope="function", params=[True, False])
def save_setup(request):
  save_all_devices = request.param
  # save model in a shared directory accessible to all ranks
  save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_save_model")
  if save_all_devices:
    save_dir = save_dir + str(tnt.get_rank())

  yield {'save_dir'    : save_dir,
         'all_devices' : save_all_devices}

  # clean up
  if save_all_devices or tnt.is_master_rank():
    shutil.rmtree(save_dir, ignore_errors=True)


def get_compile_params():
  return {'optimizer'  : keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
          'loss'       : keras.losses.SparseCategoricalCrossentropy(),
          'metrics'    :[keras.metrics.SparseCategoricalAccuracy()]}

def get_tnt_model_compiled(model):
  tnt_model = tnt.Model(model)
  tnt_model.compile(**get_compile_params())
  return tnt_model

class TestsModelLoadSave:
  model_configuration_checks = [pytest.param(util.check_model_configuration_identical,
                                             marks=[pytest.mark.min_tfversion('2.2')]),
                                pytest.param(util.check_model_configuration_identical_legacy,
                                             marks=[pytest.mark.tfversion('2.0'),
                                                    pytest.mark.tfversion('2.1')]),
                                ]
  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_save_before_compile(self, model, save_setup,
                               check_configuration_identical):
    tnt_model = tnt.Model(model)
    tnt_model.save(save_setup['save_dir'], tnt_save_all_devices = save_setup['all_devices'])
    reloaded_tnt_model = tnt.models.load_model(save_setup['save_dir'])

    assert isinstance(reloaded_tnt_model, tnt.Model)
    check_configuration_identical(reloaded_tnt_model, tnt_model)
    

  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_save_load_before_training(self, model, save_setup,
                                     check_configuration_identical):
    tnt_model = get_tnt_model_compiled(model)
    tnt_model.save(save_setup['save_dir'], tnt_save_all_devices = save_setup['all_devices'])
    reloaded_tnt_model = tnt.models.load_model(save_setup['save_dir'], compile = True)

    check_configuration_identical(reloaded_tnt_model, tnt_model)
    util.compare_weights(reloaded_tnt_model.get_weights(), tnt_model.get_weights(), 1e-6)


  @pytest.mark.parametrize("load_compiled_model", [True, False])
  def test_load_model_with_compile_flag(self, model, save_setup, load_compiled_model):
    tnt_model = get_tnt_model_compiled(model)
    train_dataset, _ = util.train_test_mnist_datasets(nbatches = 1, micro_batch_size = 32)

    tnt_model.save(save_setup['save_dir'],
                   tnt_save_all_devices = save_setup['all_devices'])
    reloaded_tnt_model = tnt.models.load_model(save_setup['save_dir'],
                                               compile = load_compiled_model)

    if not load_compiled_model:
      # if the model is not compiled, training should not succeed
      with pytest.raises(RuntimeError):
        reloaded_tnt_model.fit(train_dataset, epochs = 1, verbose = 0)
    else: # load compiled model
      # should be able to train the model if it was previously compiled
      reloaded_tnt_model.fit(train_dataset, epochs = 1, verbose = 0)


  @pytest.mark.parametrize("tf_format", [True, False])
  def test_save_weights_after_training(self, model, save_setup, tf_format):
    # create un-shuffling dataset to be able to continue training identically
    # both on the `tnt.Model` and then on the `keras.Model`
    train_dataset, _ = util.train_test_mnist_datasets(nbatches = 10, micro_batch_size = 32,
                                                      shuffle = False)
    # create and train model
    tnt_model = get_tnt_model_compiled(model)
    tnt_model.fit(train_dataset, epochs = 1, verbose = 0)

    os.makedirs(save_setup['save_dir'], exist_ok = True)
    save_path = os.path.join(save_setup['save_dir'], "weight")
    if not tf_format:
      save_path = save_path + ".h5"

    tnt_model.save_weights(save_path, tnt_save_all_devices = save_setup['all_devices'])

    # create new model with same architecture and optimizer
    if isinstance(model, tf.keras.Sequential):
      model_from_config = tnt.Sequential.from_config(tnt_model.get_config())
    elif isinstance(model, tf.keras.Model):
      model_from_config = tnt.models.model_from_config(tnt_model.get_config())

    model_from_config.compile(**get_compile_params())
    model_from_config.load_weights(save_path)
    util.compare_weights(tnt_model.get_weights(), model_from_config.get_weights(), 1e-6)

    # using the TF format saves the state together with the weights
    # such that training can continue on the loaded model
    if tf_format:
      tnt_model.fit(train_dataset, epochs = 1, verbose = 0)
      model_from_config.fit(train_dataset, epochs = 1, verbose = 0)

      util.compare_weights(tnt_model.get_weights(), model_from_config.get_weights(), 1e-6)
