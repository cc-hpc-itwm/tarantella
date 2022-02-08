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
  tnt_model_runner = base_runner.generate_tnt_model_runner(base_runner.ModelConfig(request.param,
                                                                                   tnt.ParallelStrategy.DATA))
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

remainder_samples_per_batch_list = [0, # batch size is a multiple of the number of ranks
                                    7, # some ranks have one additional sample in their micro-batch
                                    ]
last_incomplete_batch_size_list = [0, # number of samples is a multiple of batch size
                                   pytest.param(23, # last_batch_size >= number of ranks (no padding)
                                                marks=pytest.mark.min_tfversion('2.2')),
                                   pytest.param(1, # last_batch_size < number of ranks
                                                   # (i.e., padding is required)
                                                marks=pytest.mark.min_tfversion('2.2')),
                                  ]
class TestsDataParallelCompareAccuracyAnyBatchSize:
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("test_nbatches", [10])
  @pytest.mark.parametrize("remainder_samples_per_batch", remainder_samples_per_batch_list)
  @pytest.mark.parametrize("last_incomplete_batch_size", last_incomplete_batch_size_list)
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches,
                                              remainder_samples_per_batch, last_incomplete_batch_size):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(
                                    nbatches, test_nbatches, micro_batch_size,
                                    shuffle = False,
                                    remainder_samples_per_batch = remainder_samples_per_batch,
                                    last_incomplete_batch_size = last_incomplete_batch_size)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(
                                            nbatches, test_nbatches, micro_batch_size,
                                            shuffle = False,
                                            remainder_samples_per_batch = remainder_samples_per_batch,
                                            last_incomplete_batch_size = last_incomplete_batch_size)

    tnt_model_runner, reference_model_runner = model_runners
    
    ref_history = reference_model_runner.train_model(ref_train_dataset, number_epochs)
    tnt_history = tnt_model_runner.train_model(train_dataset, number_epochs)
    
    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    ref_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Reference [loss, accuracy] = {ref_loss_accuracy}")

    assert np.isclose(tnt_loss_accuracy[0], ref_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], ref_loss_accuracy[1], atol=1e-6)
