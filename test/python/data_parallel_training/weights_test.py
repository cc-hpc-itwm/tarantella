from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow.keras as keras
import pytest

# Run tests with multiple models as fixtures
@pytest.fixture(scope="function", params=[mnist.lenet5_model_generator,
                                          mnist.sequential_model_generator])
def tnt_model_runner(request):
  yield base_runner.generate_tnt_model_runner(request.param())

class TestsDataParallelCompareWeights:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("nbatches", [10])
  def test_weights_are_identical_across_ranks(self, tnt_model_runner, micro_batch_size,
                                              number_epochs, nbatches):
    (train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                        micro_batch_size = micro_batch_size)
    tnt_model_runner.train_model(train_dataset, number_epochs)
    tnt_weights = tnt_model_runner.get_weights()

    root_rank = tnt.get_size() - 1
    broadcaster = tnt.TensorBroadcaster(tnt_model_runner.get_weights(), root_rank)
    if tnt.get_rank() == root_rank:
      broadcaster.broadcast(tnt_model_runner.get_weights())
    else:
      root_rank_weights = broadcaster.broadcast()
      util.compare_weights(tnt_weights, root_rank_weights, 1e-6)
