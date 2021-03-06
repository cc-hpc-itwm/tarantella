from models import mnist_models as mnist
import utilities as util
import tarantella

import tensorflow as tf
from tensorflow import keras

import pytest

class TestsSGDMomentumOptimizer:
  @pytest.mark.parametrize("keras_model", [mnist.lenet5_model_generator,
                                           mnist.sequential_model_generator])
  @pytest.mark.parametrize("lr", [0.01])
  @pytest.mark.parametrize("nesterov", [False, True])
  @pytest.mark.parametrize("momentum", [0.9])
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  @pytest.mark.parametrize("number_epochs", [8])
  def test_compare_sgd_momentum(self, tarantella_framework, keras_model,
                                lr, nesterov, momentum, micro_batch_size, nbatches,
                                number_epochs):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    model = tarantella.Model(keras_model())
    model.compile(keras.optimizers.SGD(learning_rate=lr,
                                       momentum=momentum,
                                       nesterov=nesterov),
                  loss = keras.losses.SparseCategoricalCrossentropy(),
                  metrics = [keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_dataset,
              epochs = number_epochs,
              verbose = 0)
    results = model.evaluate(test_dataset)
    util.check_accuracy_greater(results[1], 0.91)
