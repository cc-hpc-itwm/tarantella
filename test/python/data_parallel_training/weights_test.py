from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import logging
import pytest

# Run tests with multiple models as fixtures 
# (reuse the same model for various test parameter combinations)
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.sequential_model_generator,
                                      ])
def model_runner(request):
  yield base_runner.generate_tnt_model_runner(request.param())

class TestsDataParallelCompareWeights:

  def test_initialization(self, tarantella_framework):
    assert tarantella_framework

  def test_model_initialization(self, model_runner):
    assert model_runner.model

  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [100])
  @pytest.mark.parametrize("number_epochs", [7])
  def test_compare_weights_across_ranks(self, tarantella_framework, model_runner,
                                        micro_batch_size, nbatches, number_epochs):
    comm_size = tarantella_framework.get_size()
    batch_size = micro_batch_size * comm_size
    nsamples = nbatches * batch_size

    (train_dataset, _) = util.load_dataset(mnist.load_mnist_dataset,
                                           train_size = nsamples,
                                           train_batch_size = batch_size,
                                           test_size = 0,
                                           test_batch_size = batch_size)
    model_runner.reset_weights()
    model_runner.train_model(train_dataset, number_epochs)
    final_weights = model_runner.get_weights()

    # broadcast the weights from the master rank to all the participating ranks
    model_runner.model._broadcast_weights()

    reference_rank_weights = model_runner.get_weights()
    util.compare_weights(final_weights, reference_rank_weights, 1e-6)
