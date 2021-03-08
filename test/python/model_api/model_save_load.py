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
    if tnt.is_master_rank():
      shutil.rmtree(save_dir, ignore_errors=True)


  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  @pytest.mark.parametrize("save_all_devices", [True, False])
  def test_save_load_before_training(self, tarantella_framework, model,
                                   check_configuration_identical,save_all_devices):
    tnt_model = tnt.Model(model)
    tnt_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                      loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = [keras.metrics.SparseCategoricalAccuracy()])

    # save model in a shared directory accessible to all ranks
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_save_load_before_training")
    if save_all_devices:
      save_dir = save_dir + str(tnt.get_rank())
    
    tnt_model.save(save_dir,tnt_save_all_devices = save_all_devices)

    # load into a new tnt.Model
    loaded_model = tnt.models.load_model(save_dir, compile = True)
    assert isinstance(loaded_model, tnt.Model)
    check_configuration_identical(loaded_model, tnt_model)

    util.compare_weights(loaded_model.get_weights(), tnt_model.get_weights(), 1e-6)
    
    #clean up errors is ignored
    if save_all_devices:
      shutil.rmtree(save_dir, ignore_errors=True)
    else:
      if tnt.is_master_rank():
        shutil.rmtree(save_dir, ignore_errors=True)


  @pytest.mark.parametrize("load_compiled_model", [True, False])
  @pytest.mark.parametrize("save_all_devices", [True, False])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("check_configuration_identical", model_configuration_checks)
  def test_save_load_after_training(self, tarantella_framework, model,
                                  load_compiled_model, micro_batch_size, nbatches,
                                  check_configuration_identical,save_all_devices):
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
                             "test_save_load_after_training")
    if save_all_devices:
      save_dir = save_dir + str(tnt.get_rank())
    
    tnt_model.save(save_dir,tnt_save_all_devices = save_all_devices)

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

    # cleanup, errors is ignored
    if save_all_devices:
      shutil.rmtree(save_dir, ignore_errors=True)
    else:
      if tnt.is_master_rank():
        shutil.rmtree(save_dir, ignore_errors=True)
    
    
  @pytest.mark.parametrize("save_all_devices", [True, False])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("tf_format", [True, False])
  def test_weights_after_training(self, tarantella_framework, model,
                                  micro_batch_size, nbatches,save_all_devices,
                                  tf_format):
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
    tnt_model.fit(train_dataset,
                  epochs = 3,
                  verbose = 0)
    
    #Get model_from_config with same architecture and optimizer
    model_from_config = tnt.models.model_from_config(tnt_model.get_config())
    model_from_config.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                              loss = keras.losses.SparseCategoricalCrossentropy(),
                              metrics = [keras.metrics.SparseCategoricalAccuracy()])
    
    # save model in a shared directory accessible to all ranks
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_weights_after_training")
    if save_all_devices:
      save_dir = save_dir + str(tnt.get_rank())
      os.mkdir(save_dir)
    
    if not save_all_devices:
      if tnt.is_master_rank():
        os.mkdir(save_dir)
      
    save_path = os.path.join(save_dir,"weight")
    if not tf_format:
      save_path = save_path + ".h5"
    
    tnt_model.save_weights(save_path,tnt_save_all_devices = save_all_devices)
    
    model_from_config.load_weights(save_path)
    
    util.compare_weights(tnt_model.get_weights(), model_from_config.get_weights(), 1e-6)
    
    if tf_format:
      util.compare_weights(tnt_model.orig_optimizer.get_weights()[1],model_from_config.orig_optimizer.get_weights()[0],1e-6)
      util.compare_weights(tnt_model.dist_optimizer.get_weights()[1],model_from_config.dist_optimizer.get_weights()[0],1e-6)
      tnt_model.fit(train_dataset,
                  epochs = 1,
                  verbose = 0)
      model_from_config.fit(train_dataset,
                  epochs = 1,
                  verbose = 0)
      util.compare_weights(tnt_model.get_weights(), model_from_config.get_weights(), 1e-6)
      
    
    if save_all_devices:
      shutil.rmtree(save_dir, ignore_errors=True)
    else:
      if tnt.is_master_rank():
        shutil.rmtree(save_dir, ignore_errors=True)
    
    
    
    
    
    
    
