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

class TestCloneModel:
  def test_clone_model(self, tarantella_framework, keras_model):
    cloned_model = tnt.models.clone_model(keras_model)
    tnt_model = tnt.Model(keras_model)
    util.check_model_configuration_identical(tnt_model, cloned_model)

