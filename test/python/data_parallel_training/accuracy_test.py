from models import mnist_models as mnist
import training_runner
import utilities as util
import tarantella as tnt

import numpy as np
import logging
import pytest

# Run tests with multiple models as fixtures
@pytest.fixture(scope="function", params=[training_runner.ModelConfig(mnist.fc_model_generator),
                                          pytest.param(training_runner.ModelConfig(mnist.fc_model_generator_two_partitions,
                                                                                   tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 2,
                                                                                reason="Not enough ranks for two partitions")),
                                          training_runner.ModelConfig(mnist.lenet5_model_generator),
                                          training_runner.ModelConfig(mnist.sequential_model_generator),
                                          training_runner.ModelConfig(mnist.subclassed_model_generator)
                                          ])
def model_runners(request):
  tnt_model_runner = training_runner.generate_tnt_model_runner(request.param)
  reference_model_runner = training_runner.TrainingRunner(request.param.model_generator())
  yield tnt_model_runner, reference_model_runner

class TestsDataParallelCompareAccuracy:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [3])
  @pytest.mark.parametrize("nbatches", [20])
  @pytest.mark.parametrize("test_nbatches", [5])
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches):
    (train_dataset, test_dataset) = util.train_test_mnist_datasets(nbatches = nbatches, test_nbatches = test_nbatches,
                                                                   micro_batch_size = micro_batch_size,
                                                                   drop_remainder = True)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches = nbatches, test_nbatches = test_nbatches,
                                                                           micro_batch_size = micro_batch_size,
                                                                           drop_remainder = True)
    tnt_model_runner, reference_model_runner = model_runners

    tnt_model_runner.train_model(train_dataset, number_epochs)
    reference_model_runner.train_model(ref_train_dataset, number_epochs)

    tnt_loss_accuracy = tnt_model_runner.evaluate_model(test_dataset)
    reference_loss_accuracy = reference_model_runner.evaluate_model(ref_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Reference [loss, accuracy] = {reference_loss_accuracy}")

    result = [True, True]
    if tnt.is_master_rank():
      result = [np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2), # losses might not be identical
                np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)]
    util.assert_on_all_ranks(result)

