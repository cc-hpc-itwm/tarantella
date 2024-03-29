from models import mnist_models as mnist
import utilities as util
import tarantella as tnt
import tensorflow
import pytest
import yaml

# saving/loading architecture only applies to keras models or Sequential
# (https://www.tensorflow.org/guide/keras/save_and_serialize#saving_the_architecture)
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.alexnet_model_generator])
def model(request):
  yield request.param()

@pytest.mark.parametrize("parallel_strategy", [tnt.ParallelStrategy.DATA,
                                               tnt.ParallelStrategy.ALL])
class TestModelSaveLoadInMemory:
  def test_model_get_config(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    assert tnt_model.get_config() == model.get_config()

  def test_model_to_json(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    assert tnt_model.to_json() == model.to_json()

  @pytest.mark.max_tfversion('2.5') # method deprecated later
  def test_model_to_yaml(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    assert tnt_model.to_yaml() == model.to_yaml()

  def test_model_from_config(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    config = tnt_model.get_config()
    model_from_config = tnt.models.model_from_config(config)
    util.check_model_configuration_identical(model_from_config, model)

  def test_model_from_json(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    json = tnt_model.to_json()
    json_model = tnt.models.model_from_json(json)
    util.check_model_configuration_identical(json_model, model)

  @pytest.mark.max_tfversion('2.5') # method deprecated later
  def test_model_from_yaml(self, model, parallel_strategy):
    tnt_model = tnt.Model(model, parallel_strategy)
    yaml = tnt_model.to_yaml()
    yaml_model = tnt.models.model_from_yaml(yaml)
    util.check_model_configuration_identical(yaml_model, model)


@pytest.fixture(scope="class", params=[mnist.sequential_model_generator])
def sequential(request):
  yield request.param()

@pytest.mark.parametrize("parallel_strategy", [tnt.ParallelStrategy.DATA,
                                               tnt.ParallelStrategy.ALL,])
class TestSequentialSaveLoadInMemory:
  def test_model_from_config(self, sequential, parallel_strategy):
    tnt_model = tnt.Model(sequential, parallel_strategy)
    config = tnt_model.get_config()
    model_from_config = tnt.Sequential.from_config(config)
    util.check_model_configuration_identical(model_from_config, sequential)
