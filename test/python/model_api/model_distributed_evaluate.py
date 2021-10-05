from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt
import tensorflow as tf

import numpy as np

import logging
import pytest

@pytest.fixture(scope="function", params=[mnist.fc_model_generator,
                                          mnist.lenet5_model_generator,
                                          mnist.sequential_model_generator,
                                          mnist.subclassed_model_generator
                                         ])

def model_runners(request):
#   tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield reference_model_runner

class TestsDistributedEvaluation:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [4])
  @pytest.mark.parametrize("nbatches", [8])
  @pytest.mark.parametrize("test_nbatches", [10])
  @pytest.mark.parametrize("extra_batch", [0, 5])
  @pytest.mark.parametrize("extra_sample", [0, 33])
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches,
                                              extra_batch, extra_sample):
    (_, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                       micro_batch_size,
                                                       shuffle = False,
                                                       remainder_samples_per_batch = extra_batch,
                                                       last_incomplete_batch_size = extra_sample)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           shuffle = False,
                                                                           remainder_samples_per_batch = extra_batch,
                                                                           last_incomplete_batch_size = extra_sample)
    
    reference_model_runner = model_runners
    reference_model_runner.train_model(ref_train_dataset, number_epochs)
    tnt_cloned_model = base_runner.generate_tnt_model_runner(reference_model_runner.model)
    
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)
    tnt_loss_accuracy = tnt_cloned_model.evaluate_model(test_dataset, distribution = True)
    
    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Reference [loss, accuracy] = {reference_loss_accuracy}")
    assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2)
    assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)
