from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella

import tensorflow as tf
from tensorflow import keras

import pytest

# Run tests with multiple models as fixtures
# (reuse the same model for various test parameter combinations)
# Fixture for MNIST models
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.sequential_model_generator
                                      ])
def mnist_model_runner(request):
  yield base_runner.generate_tnt_model_runner(request.param())

class TestsDataParallelOptimizers:
  def test_initialization(self, tarantella_framework):
    assert tarantella_framework

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
  def test_compare_accuracy_optimizers(self, tarantella_framework, mnist_model_runner,
                                      optimizer, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (number_epochs, lr) = mnist.get_hyperparams(optimizer)
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    mnist_model_runner.compile_model(optimizer(learning_rate=lr))
    mnist_model_runner.reset_weights()
    mnist_model_runner.train_model(train_dataset, number_epochs)

    results = mnist_model_runner.evaluate_model(test_dataset)
    util.check_accuracy_greater(results[1], 0.91)

  @pytest.mark.parametrize("lr", [0.01])
  @pytest.mark.parametrize("nesterov", [False, True])
  @pytest.mark.parametrize("momentum", [0.9])
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  @pytest.mark.parametrize("number_epochs", [8])
  def test_compare_sgd_momentum(self, tarantella_framework, mnist_model_runner,
                                lr, nesterov, momentum, micro_batch_size, nbatches,
                                number_epochs):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    mnist_model_runner.compile_model(keras.optimizers.SGD(learning_rate=lr,
                                                          momentum=momentum,
                                                          nesterov=nesterov))
    mnist_model_runner.reset_weights()
    mnist_model_runner.train_model(train_dataset, number_epochs)

    results = mnist_model_runner.evaluate_model(test_dataset)
    util.check_accuracy_greater(results[1], 0.91)
