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
@pytest.fixture(scope="class", params=[mnist.sequential_model_generator])
def model(request):
  yield request.param()


@pytest.fixture(scope="function", params=[False])
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
  return {'loss'       : keras.losses.SparseCategoricalCrossentropy(),
          'metrics'    :[keras.metrics.SparseCategoricalAccuracy()]}

def get_tnt_model_compiled(model, parallel_strategy, optimizer):
  tnt_model = tnt.Model(model, parallel_strategy)
  tnt_model.compile(optimizer = optimizer, **get_compile_params())
  return tnt_model

class TestsModelLoadSaveState:

  model_configuration_checks = [pytest.param(util.check_model_configuration_identical,
                                             marks=[pytest.mark.min_tfversion('2.2')]),
                                pytest.param(util.check_model_configuration_identical_legacy,
                                             marks=[pytest.mark.tfversion('2.0'),
                                                    pytest.mark.tfversion('2.1')]),
                                ]

  # Example from TensorFlow model saving guide
  # https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading
  @pytest.mark.xfail
  @pytest.mark.parametrize("optimizer", [keras.optimizers.SGD,
                                         keras.optimizers.Adam,
                                         keras.optimizers.RMSprop])
  def test_simple_keras_models(self, save_setup, optimizer):
    def get_model():
      # Create a simple model.
      inputs = keras.Input(shape=(32,))
      outputs = keras.layers.Dense(1)(inputs)
      model = keras.Model(inputs, outputs)
      model.compile(optimizer=optimizer(), loss="mean_squared_error")
      return model

    tf.random.set_seed(42)
    model = get_model()

    # Train the model.
    test_input = np.ones((128, 32))
    test_target = np.ones((128, 1))
    model.fit(test_input, test_target)
    model.save(save_setup['save_dir'])

    tf.random.set_seed(42)
    reconstructed_model = keras.models.load_model(save_setup['save_dir'])

    # Let's check:
    np.testing.assert_allclose(model.predict(test_input),
                               reconstructed_model.predict(test_input))
    util.compare_weights(reconstructed_model.get_weights(), model.get_weights(), 1e-6)

    tf.random.set_seed(42)
    reconstructed_model.fit(test_input, test_target, epochs = 3, shuffle = False)
    tf.random.set_seed(42)
    model.fit(test_input, test_target, epochs = 3, shuffle = False)
    util.compare_weights(model.get_weights(), reconstructed_model.get_weights(), 1e-6)


  @pytest.mark.xfail
  @pytest.mark.parametrize("optimizer", [keras.optimizers.SGD,
                                         keras.optimizers.Adam,
                                         keras.optimizers.RMSprop])
  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_keras_models(self, model, save_setup, optimizer, check_configuration_identical):
    train_dataset, _ = util.train_test_mnist_datasets(nbatches = 10, micro_batch_size = 32,
                                                      shuffle = False)
    # train model
    keras_model = model
    keras_model.compile(optimizer(),
                        loss = keras.losses.SparseCategoricalCrossentropy(),
                        metrics = [keras.metrics.SparseCategoricalAccuracy()])
    keras_model.fit(train_dataset, epochs = 2, verbose = 0)
    keras_model.save(save_setup['save_dir'])

    reloaded_model = keras.models.load_model(save_setup['save_dir'])
    check_configuration_identical(reloaded_model, keras_model)

    # continue training on the original model
    keras_model.fit(train_dataset, epochs = 2, shuffle = False, verbose = 0)

    # continue training on the loaded model
    reloaded_model.fit(train_dataset, epochs = 2, shuffle = False, verbose = 0)

    util.compare_weights(reloaded_model.get_weights(), keras.get_weights(), 1e-6)


  @pytest.mark.xfail
  @pytest.mark.parametrize("parallel_strategy", [tnt.ParallelStrategy.DATA,
                                                 pytest.param(tnt.ParallelStrategy.ALL, marks=pytest.mark.xfail),])
  @pytest.mark.parametrize("optimizer_type", [keras.optimizers.SGD,
                                              keras.optimizers.Adam,
                                              keras.optimizers.RMSprop])
  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_save_load_train_models(self, model, save_setup, parallel_strategy, optimizer_type,
                                  check_configuration_identical):
    train_dataset, _ = util.train_test_mnist_datasets(nbatches = 10, micro_batch_size = 32,
                                                      shuffle = False)
    # create and train model
    tnt_model = get_tnt_model_compiled(model, parallel_strategy, optimizer_type())
    tnt_model.fit(train_dataset, epochs = 2, verbose = 0)
    tnt_model.save(save_setup['save_dir'], tnt_save_all_devices = save_setup['all_devices'])

    # load into a new tnt.Model
    reloaded_tnt_model = tnt.models.load_model(save_setup['save_dir'])
    assert isinstance(reloaded_tnt_model, tnt.Model)
    check_configuration_identical(reloaded_tnt_model, tnt_model)

    # continue training on the original model
    tnt_model.fit(train_dataset, epochs = 2, verbose = 0)

    # continue training on the loaded model
    reloaded_tnt_model.fit(train_dataset, epochs = 2, verbose = 0)

    util.compare_weights(reloaded_tnt_model.get_weights(), tnt_model.get_weights(), 1e-6)
