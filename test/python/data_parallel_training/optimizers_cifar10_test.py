from models import cifar10_models as cifar
import training_runner as base_runner
import utilities as util
import tarantella

import tensorflow as tf
from tensorflow import keras

import pytest

# Fixture for CIFAR-10 models
@pytest.fixture(scope="class", params=[cifar.alexnet_model_generator])
def cifar_model_runner(request):
  yield base_runner.generate_tnt_model_runner(request.param())

class TestsDataParallelOptimizersCIFAR10:
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
  @pytest.mark.parametrize("ntest_batches", [40])
  def test_cifar_alexnet(self, tarantella_framework, cifar_model_runner,
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
      cifar_model_runner.compile_model(optimizer(learning_rate=lr, momentum=0.9))
    else:
      cifar_model_runner.compile_model(optimizer(learning_rate=lr))

    cifar_model_runner.reset_weights()
    cifar_model_runner.train_model(train_dataset, number_epochs)

    results = cifar_model_runner.evaluate_model(test_dataset)
    util.check_accuracy_greater(results[1], 0.5)
