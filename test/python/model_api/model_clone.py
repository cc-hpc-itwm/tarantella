from models import mnist_models as mnist
import utilities as util
import tarantella as tnt

import pytest

@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.alexnet_model_generator,
                                       mnist.sequential_model_generator
                                       ])
def keras_model(request):
  yield request.param()

@pytest.mark.parametrize("enable_model_parallelism", [False,
                                                      pytest.param(True, marks=pytest.mark.xfail),])
class TestCloneModel:
  def test_clone_keras_model(self, keras_model, enable_model_parallelism):
    cloned_model = tnt.models.clone_model(keras_model)
    tnt_model = tnt.Model(keras_model, enable_model_parallelism = enable_model_parallelism)
    util.check_model_configuration_identical(tnt_model, cloned_model)

  def test_clone_tnt_model(self, keras_model, enable_model_parallelism):
    tnt_model = tnt.Model(keras_model, enable_model_parallelism = enable_model_parallelism)
    cloned_model = tnt.models.clone_model(tnt_model)
    util.check_model_configuration_identical(tnt_model, cloned_model)
