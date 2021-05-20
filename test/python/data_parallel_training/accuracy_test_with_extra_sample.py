from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt
import tensorflow as tf

import numpy as np

import logging
import pytest

# Run tests with multiple models as fixtures
@pytest.fixture(scope="function", params=[mnist.fc_model_generator
                                         ])
def model_runners(request):
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

class TestsDataParallelCompareAccuracy:
  @pytest.mark.skipif(tf.version.VERSION < "2.2.0",reason="requires tf >= 2.2")
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("number_epochs", [4])
  @pytest.mark.parametrize("nbatches", [15])
  @pytest.mark.parametrize("test_nbatches", [2])
  @pytest.mark.parametrize("extra_batch", [0, 5, 7])
  @pytest.mark.parametrize("extra_sample", [5, 7])
  def test_compare_accuracy_against_reference_with_pad(self, model_runners, micro_batch_size,
                                                           number_epochs, nbatches, test_nbatches,
                                                           extra_batch, extra_sample):
    (train_dataset, _) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size,
                                                                   extra_batch = extra_batch,
                                                                   extra_sample = extra_sample)
    (ref_train_dataset, _) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           extra_batch = extra_batch,
                                                                           extra_sample = extra_sample)

    tnt_model_runner, reference_model_runner = model_runners
    
    ref_history = reference_model_runner.train_model(ref_train_dataset, number_epochs)
    tnt_history = tnt_model_runner.train_model(train_dataset, number_epochs)
    
    tolerate = 1e-2
    
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'], tolerate)
    assert np.allclose(tnt_history.history['sparse_categorical_accuracy'], ref_history.history['sparse_categorical_accuracy'], tolerate)