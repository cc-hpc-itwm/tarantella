from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella


import tensorflow as tf
from tensorflow import keras
import numpy as np

import logging
import pytest

# Run tests with multiple models as fixtures
# (reuse the same model for various test parameter combinations)
@pytest.fixture(scope="class", params=[mnist.fc_model_generator,
                                       mnist.lenet5_model_generator,
                                       mnist.sequential_model_generator,
                                       mnist.subclassed_model_generator,
                                      ])
def model_runners(request):
  tf.random.set_seed(42)
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  tf.random.set_seed(42)
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

class TestsDataParallelCompareAccuracy:

  def test_initialization(self, tarantella_framework):
    assert tarantella_framework

  @pytest.mark.parametrize("micro_batch_size", [32, 61])
  @pytest.mark.parametrize("number_epochs", [3])
  @pytest.mark.parametrize("nbatches", [200])
  def test_compare_accuracy_against_reference(self, tarantella_framework, model_runners,
                                              micro_batch_size, number_epochs, nbatches):
    batch_size = micro_batch_size * tarantella_framework.get_size()
    nsamples = nbatches * batch_size

    tnt_model_runner, reference_model_runner = model_runners
    # reuse model with its initial weights
    tnt_model_runner.reset_weights()
    reference_model_runner.reset_weights()

    # verify that both models have identical weights
    tnt_initial_weights = tnt_model_runner.get_weights()
    reference_initial_weights = reference_model_runner.get_weights()
    util.compare_weights(tnt_initial_weights, reference_initial_weights, 1e-6)

    # train reference model
    (ref_train_dataset, ref_test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                              train_size = nsamples,
                                                              train_batch_size = batch_size,
                                                              test_size = 10000,
                                                              test_batch_size = batch_size)
    reference_model_runner.train_model(ref_train_dataset, number_epochs)
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    # train Tarantella model
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    tnt_model_runner.train_model(train_dataset, number_epochs)
    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)

    rank = tarantella_framework.get_rank()
    logging.getLogger().info("[Rank %d] Tarantella[loss, accuracy] = %s" % (rank, str(tnt_loss_accuracy)))
    logging.getLogger().info("[Rank %d] Reference [loss, accuracy] = %s" % (rank, str(reference_loss_accuracy)))
    assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-2)
