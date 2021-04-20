from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import pytest

# Run tests with multiple models as fixtures
@pytest.fixture(scope="function", params=[mnist.lenet5_model_generator,
                                          mnist.sequential_model_generator])
def model_runners(request):
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

class TestsDataParallelCompareWeights:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("nbatches", [10])
  def test_compare_weights_across_ranks(self, model_runners, micro_batch_size,
                                        number_epochs, nbatches):
    (train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                        micro_batch_size = micro_batch_size)
    (ref_train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                            micro_batch_size = micro_batch_size)

    tnt_model_runner, reference_model_runner = model_runners

    tnt_model_runner.train_model(train_dataset, number_epochs)
    reference_model_runner.train_model(ref_train_dataset, number_epochs)

    tnt_weights = tnt_model_runner.get_weights()
    reference_weights = reference_model_runner.get_weights()
    util.compare_weights(tnt_weights, reference_weights, 1e-4)
