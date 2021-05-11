from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt
import tensorflow as tf

import numpy as np

import logging
import pytest

# Run tests with multiple models as fixtures
@pytest.fixture(scope="function", params=[mnist.fc_model_generator,
                                          mnist.lenet5_model_generator,
                                          mnist.sequential_model_generator,
                                          mnist.subclassed_model_generator])
def model_runners(request):
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

class TestsDataParallelCompareAccuracy:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [3])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("test_nbatches", [2])
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size)

    tnt_model_runner, reference_model_runner = model_runners
    tnt_model_runner.train_model(train_dataset, number_epochs)
    reference_model_runner.train_model(ref_train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Reference [loss, accuracy] = {reference_loss_accuracy}")
    assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)
  
  @pytest.mark.skipif(tf.version.VERSION >= "2.2.0",reason="requires tf < 2.2")
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("number_epochs", [4])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("test_nbatches", [2])
  @pytest.mark.parametrize("extra_batch", [5, 7, 19])
  def test_compare_accuracy_against_reference_with_diff_micro_batch(self, model_runners, micro_batch_size,
                                                           number_epochs, nbatches, test_nbatches,
                                                           extra_batch):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size,
                                                                   extra_batch = extra_batch)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           extra_batch = extra_batch,drop_remainder = True)

    tnt_model_runner, reference_model_runner = model_runners
    reference_model_runner.train_model(ref_train_dataset, number_epochs)
    tnt_model_runner.train_model(train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info("[Rank %d] Tarantella[loss, accuracy] = %s" % (rank, str(tnt_loss_accuracy)))
    logging.getLogger().info("[Rank %d] Reference [loss, accuracy] = %s" % (rank, str(reference_loss_accuracy)))
    assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)
    
  @pytest.mark.skipif(tf.version.VERSION < "2.2.0",reason="requires tf >= 2.2")
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("number_epochs", [4])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("test_nbatches", [2])
  @pytest.mark.parametrize("extra_batch", [5, 11, 17])
  @pytest.mark.parametrize("extra_sample", [0])
  def test_compare_accuracy_against_reference_with_pad(self, model_runners, micro_batch_size,
                                                           number_epochs, nbatches, test_nbatches,
                                                           extra_batch, extra_sample):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size,
                                                                   extra_batch = extra_batch,
                                                                   extra_sample = extra_sample)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           extra_batch = extra_batch,
                                                                           extra_sample = extra_sample)

    tnt_model_runner, reference_model_runner = model_runners
    reference_model_runner.train_model(ref_train_dataset, number_epochs)
    tnt_model_runner.train_model(train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info("[Rank %d] Tarantella[loss, accuracy] = %s" % (rank, str(tnt_loss_accuracy)))
    logging.getLogger().info("[Rank %d] Reference [loss, accuracy] = %s" % (rank, str(reference_loss_accuracy)))

    assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)
