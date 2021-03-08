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

# saving/loading the whole model applies to keras models, subclassed models, or Sequential
# (https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading)
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.sequential_model_generator])
def model(request):
 yield request.param()

class TestsModelLoadSave:

  model_configuration_checks = [pytest.param(util.check_model_configuration_identical,
                                             marks=pytest.mark.tfversion('2.2')),
                                pytest.param(util.check_model_configuration_identical_legacy,
                                             marks=[pytest.mark.tfversion('2.0'),
                                                    pytest.mark.tfversion('2.1')]),
                                ]

  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_save_before_compile(self, tarantella_framework, model,
                               check_configuration_identical):
    tnt_model = tnt.Model(model)

    # save model in a shared directory accessible to all ranks
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_save_before_compile")
    tnt_model.save(save_dir)
    assert os.path.exists(save_dir)

    # make sure the original model has the same weights on all ranks
    tnt_model._broadcast_weights_if_necessary()

    # load file into a new Tarantella model
    loaded_model = tnt.models.load_model(save_dir, compile=True)
    assert isinstance(loaded_model, tnt.Model)

    # check whether the weights of the two models match on each rank
    check_configuration_identical(loaded_model, tnt_model)
    
    # cleanup
    if tnt.get_rank() == 0:
      shutil.rmtree(save_dir, ignore_errors=True)


  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_weights_before_training(self, tarantella_framework, model,
                                   check_configuration_identical):
    tnt_model = tnt.Model(model)
    tnt_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                      loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = [keras.metrics.SparseCategoricalAccuracy()])

    # save model in a shared directory accessible to all ranks
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_weights_before_training")
    tnt_model.save(save_dir)

    # load into a new tnt.Model
    loaded_model = tnt.models.load_model(save_dir, compile = True)
    assert isinstance(loaded_model, tnt.Model)
    check_configuration_identical(loaded_model, tnt_model)

    util.compare_weights(loaded_model.get_weights(), tnt_model.get_weights(), 1e-6)

    # cleanup
    if tnt.get_rank() == 0:
      shutil.rmtree(save_dir, ignore_errors=True)


  @pytest.mark.parametrize("load_compiled_model", [True, False])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_weights_after_training(self, tarantella_framework, model,
                                  load_compiled_model, micro_batch_size, nbatches,
                                  check_configuration_identical):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (train_dataset, _) = util.load_dataset(mnist.load_mnist_dataset,
                                           train_size = nsamples,
                                           train_batch_size = batch_size,
                                           test_size = 0,
                                           test_batch_size = batch_size)
    
    # train model
    tnt_model = tnt.Model(model)
    tnt_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                      loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = [keras.metrics.SparseCategoricalAccuracy()])

    # save model in a shared directory accessible to all ranks
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_weights_after_training")
    tnt_model.save(save_dir)

    # train original model
    tnt_model.fit(train_dataset,
                  epochs = 3,
                  verbose = 0)

    # load into a new tnt.Model
    loaded_model = tnt.models.load_model(save_dir, compile = load_compiled_model)
    assert isinstance(loaded_model, tnt.Model)
    check_configuration_identical(loaded_model, tnt_model)

    # compile model
    if not load_compiled_model:
      loaded_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                           loss = keras.losses.SparseCategoricalCrossentropy(),
                           metrics = [keras.metrics.SparseCategoricalAccuracy()])

    # train loaded model
    loaded_model.fit(train_dataset,
                     epochs = 3,
                     verbose = 0)

    util.compare_weights(loaded_model.get_weights(), tnt_model.get_weights(), 1e-6)

    # cleanup
    if tnt.get_rank() == 0:
      shutil.rmtree(save_dir, ignore_errors=True)
