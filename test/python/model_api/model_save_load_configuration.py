from models import mnist_models as mnist
import utilities as util
import tarantella as tnt
import tensorflow
import pytest
import yaml

# saving/loading architecture only applies to keras models or Sequential
# (https://www.tensorflow.org/guide/keras/save_and_serialize#saving_the_architecture)
@pytest.fixture(scope="class", params=[mnist.lenet5_model_generator,
                                       mnist.alexnet_model_generator,
                                       mnist.sequential_model_generator
                                       ])
def model(request):
  yield request.param()

class TestsModelSaveLoadInMemory:
  def test_model_get_config(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    assert tnt_model.get_config() == model.get_config()

  def test_model_to_json(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    assert tnt_model.to_json() == model.to_json()

  def test_model_to_yaml(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    assert tnt_model.to_yaml() == model.to_yaml()

  def test_model_from_config(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    config = tnt_model.get_config()

    model_from_config = tnt.models.model_from_config(config)
    assert isinstance(model_from_config, tnt.Model)
    assert util.is_model_configuration_identical(model_from_config, model)

  def test_model_from_json(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    json = tnt_model.to_json()

    json_model = tnt.models.model_from_json(json)
    assert isinstance(json_model, tnt.Model)
    assert util.is_model_configuration_identical(json_model, model)

  def test_model_from_yaml(self, tarantella_framework, model):
    tnt_model = tnt.Model(model)
    yaml = tnt_model.to_yaml()

    yaml_model = tnt.models.model_from_yaml(yaml)
    assert isinstance(yaml_model, tnt.Model)
    assert util.is_model_configuration_identical(yaml_model, model)
