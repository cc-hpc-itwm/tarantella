from models import mnist_models as mnist
import utilities as util

import tarantella as tnt
import tensorflow as tf
from tensorflow import keras

import copy
import numpy as np
import os
import pytest
import tempfile

# saving/loading the whole model applies to keras models, subclassed models, or Sequential
# (https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading)
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.sequential_model_generator
                                      ])
def model(request):
  yield request.param()

class TestsModelLoadSave:

  def test_save_before_compile(self, tarantella_framework, model):
    with tempfile.TemporaryDirectory() as tmpdirname:
      file_path = os.path.join(tmpdirname, "model.save")

      tnt_model = tnt.Model(model)

      # save Tarantella model
      assert not os.path.exists(file_path)
      tnt_model.save(file_path)
      assert os.path.exists(file_path)

      # load file into a new Tarantella model
      loaded_model = tnt.models.load_model(file_path,compile=True)
      assert isinstance(loaded_model, tnt.Model)

      assert util.is_model_configuration_identical(loaded_model, tnt_model)

  @pytest.mark.parametrize("load_compiled_model", [True, False])
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [12])
  def test_accuracy_loaded_model(self, tarantella_framework, model,
                                 load_compiled_model, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 128,
                                                      test_batch_size = batch_size)
    
    # train model
    tnt_model = tnt.Model(model)
    tnt_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                      loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = [keras.metrics.SparseCategoricalAccuracy()])
    tnt_model.fit(train_dataset,
                  epochs = 3,
                  verbose = 0)
    result = tnt_model.evaluate(test_dataset,
                                verbose = 0)

    with tempfile.TemporaryDirectory() as tmpdirname:
      file_path = os.path.join(tmpdirname, "model.save")

      # save trained model
      tnt_model.save(file_path)

      # load into a new tnt.Model
      loaded_model = tnt.models.load_model(file_path, compile = load_compiled_model)
      assert isinstance(loaded_model, tnt.Model)
      assert util.is_model_configuration_identical(loaded_model, tnt_model)

      if not load_compiled_model:
        loaded_model.compile(keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
                             loss = keras.losses.SparseCategoricalCrossentropy(),
                             metrics = [keras.metrics.SparseCategoricalAccuracy()])

      # the loaded model should provide the same accuracy
      result_loaded = loaded_model.evaluate(test_dataset,
                                            verbose = 0)
      assert np.isclose(result[0], result_loaded[0], atol=1e-6)

