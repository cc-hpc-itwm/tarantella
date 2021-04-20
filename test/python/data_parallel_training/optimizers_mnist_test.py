from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow.keras as keras
import numpy as np

import logging
import pytest

def model_runners(model_generator):
  tnt_model_runner = base_runner.generate_tnt_model_runner(model_generator())
  reference_model_runner = base_runner.TrainingRunner(model_generator())
  return tnt_model_runner, reference_model_runner

def get_compiled_models(model_generator, optimizer, **optimizer_kwargs):
  tnt_model_runner, ref_model_runner = model_runners(model_generator)
  tnt_model_runner.compile_model(optimizer(**optimizer_kwargs))
  ref_model_runner.compile_model(optimizer(**optimizer_kwargs))

  return tnt_model_runner, ref_model_runner


class TestsDataParallelOptimizers:
  @pytest.mark.parametrize("model_generator", [mnist.lenet5_model_generator,
                                               mnist.subclassed_model_generator])
  @pytest.mark.parametrize("optimizer", [keras.optimizers.Adadelta,
                                         keras.optimizers.Adagrad,
                                         keras.optimizers.Adam,
                                         keras.optimizers.Adamax,
                                         #keras.optimizers.Nadam,
                                         keras.optimizers.RMSprop,
                                         keras.optimizers.SGD,
                                        ])
  @pytest.mark.parametrize("micro_batch_size", [42])
  @pytest.mark.parametrize("nbatches", [8])
  @pytest.mark.parametrize("test_nbatches", [3])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_optimizers_compare_to_reference(self, model_generator, optimizer, micro_batch_size,
                                           nbatches, test_nbatches, number_epochs):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size)
    tnt_model_runner, ref_model_runner = get_compiled_models(model_generator, optimizer)

    tnt_model_runner.train_model(train_dataset, number_epochs)
    ref_model_runner.train_model(ref_train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    ref_loss_accuracy = ref_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info("[Rank %d] Tarantella[loss, accuracy] = %s" % (rank, str(tnt_loss_accuracy)))
    logging.getLogger().info("[Rank %d] Reference [loss, accuracy] = %s" % (rank, str(ref_loss_accuracy)))
    assert np.isclose(tnt_loss_accuracy[0], ref_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], ref_loss_accuracy[1],
                      atol=1e-2 if isinstance(optimizer(),  keras.optimizers.RMSprop) else 1e-6)

  @pytest.mark.parametrize("model_generator", [mnist.lenet5_model_generator,
                                               mnist.sequential_model_generator])
  @pytest.mark.parametrize("nesterov", [False, True])
  @pytest.mark.parametrize("momentum", [0.9])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [5])
  @pytest.mark.parametrize("test_nbatches", [3])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_sgd_momentum_compare_to_reference(self, model_generator, nesterov, momentum,
                                             micro_batch_size, nbatches, test_nbatches, number_epochs):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size)
    optimizer = keras.optimizers.SGD
    optimizer_kwargs = {'learning_rate' : 0.01,
                        'momentum' : momentum,
                        'nesterov' : nesterov}
    tnt_model_runner, ref_model_runner = get_compiled_models(model_generator, optimizer,
                                                             **optimizer_kwargs)
    tnt_model_runner.train_model(train_dataset, number_epochs)
    ref_model_runner.train_model(ref_train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    ref_loss_accuracy = ref_model_runner.evaluate_model(ref_test_dataset)

    assert np.isclose(tnt_loss_accuracy[0], ref_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], ref_loss_accuracy[1], atol=1e-6)
