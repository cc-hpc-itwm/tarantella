from models import cifar10_models as cifar
import utilities as util
import tarantella

import tensorflow as tf
from tensorflow import keras

import pytest

class TestsDataParallelOptimizersCIFAR10:
  def test_initialization(self, tarantella_framework):
    assert tarantella_framework

  @pytest.mark.parametrize("keras_model", [cifar.alexnet_model_generator])
  @pytest.mark.parametrize("optimizer", [keras.optimizers.Adadelta,
                                         keras.optimizers.Adagrad,
                                         keras.optimizers.Adam,
                                         keras.optimizers.Adamax,
                                         keras.optimizers.Nadam,
                                         keras.optimizers.RMSprop,
                                         keras.optimizers.SGD
                                        ])
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  @pytest.mark.parametrize("ntest_batches", [40])
  def test_cifar_alexnet(self, tarantella_framework, keras_model,
                         optimizer, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (number_epochs, lr) = cifar.get_hyperparams(optimizer)
    (train_dataset, test_dataset) = util.load_dataset(cifar.load_cifar_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    if optimizer.__name__ == 'SGD':
      keras_optimizer = optimizer(learning_rate=lr, momentum=0.9)
    else:
      keras_optimizer = optimizer(learning_rate=lr)

    model = tarantella.Model(keras_model())
    model.compile(keras_optimizer,
                  loss = keras.losses.SparseCategoricalCrossentropy(),
                  metrics = [keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_dataset,
              epochs = number_epochs,
              verbose = 0)
    results = model.evaluate(test_dataset)
    util.check_accuracy_greater(results[1], 0.5)
