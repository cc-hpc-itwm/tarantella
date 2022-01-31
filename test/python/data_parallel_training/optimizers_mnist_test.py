from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow.keras as keras
import numpy as np

import logging
import pytest

metric = 'sparse_categorical_accuracy'

def model_runners(model_config):
  tnt_model_runner = base_runner.generate_tnt_model_runner(model_config)
  reference_model_runner = base_runner.TrainingRunner(model_config.model_generator())
  return tnt_model_runner, reference_model_runner

def get_compiled_models(model_config, optimizer, **optimizer_kwargs):
  tnt_model_runner, ref_model_runner = model_runners(model_config)
  tnt_model_runner.compile_model(optimizer(**optimizer_kwargs))
  ref_model_runner.compile_model(optimizer(**optimizer_kwargs))

  return tnt_model_runner, ref_model_runner

def train_tnt_and_reference_models(model_config, optimizer, micro_batch_size,
                                   nbatches, number_epochs, optimizer_kwargs = {}):
  (train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                      micro_batch_size = micro_batch_size)
  (ref_train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                          micro_batch_size = micro_batch_size)
  tnt_model_runner, ref_model_runner = get_compiled_models(model_config, optimizer,
                                                           **optimizer_kwargs)

  tnt_history = tnt_model_runner.train_model(train_dataset, number_epochs)
  ref_history = ref_model_runner.train_model(ref_train_dataset, number_epochs)

  rank = tnt.get_rank()
  logging.getLogger().info(f"[Rank {rank}] Tarantella (loss, accuracy) = "
                            f"({tnt_history.history['loss']}, {tnt_history.history[metric]})")
  logging.getLogger().info(f"[Rank {rank}] Reference (loss, accuracy) = "
                            f"({ref_history.history['loss']}, {ref_history.history[metric]})")
  return tnt_history, ref_history

class TestsDataParallelOptimizers:

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.lenet5_model_generator, False),
                                            pytest.param(base_runner.ModelConfig(mnist.lenet5_model_generator, True),
                                                         marks=pytest.mark.xfail),
                                            base_runner.ModelConfig(mnist.subclassed_model_generator)])
  @pytest.mark.parametrize("optimizer", [keras.optimizers.Adadelta,
                                         keras.optimizers.Adagrad,
                                         keras.optimizers.SGD])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [15])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_optimizers_compare_to_reference(self, model_config, optimizer, micro_batch_size,
                                           nbatches, number_epochs):
    tnt_history, ref_history = train_tnt_and_reference_models(model_config, optimizer,
                                             micro_batch_size, nbatches, number_epochs)
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'], 1e-4)
    assert np.allclose(tnt_history.history[metric], ref_history.history[metric], 1e-6)

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.lenet5_model_generator, False),
                                            pytest.param(base_runner.ModelConfig(mnist.lenet5_model_generator, True),
                                                         marks=pytest.mark.xfail),
                                            base_runner.ModelConfig(mnist.subclassed_model_generator)])
  @pytest.mark.parametrize("optimizer", [keras.optimizers.Adam,
                                         keras.optimizers.Adamax,
                                         #keras.optimizers.Nadam,
                                         keras.optimizers.RMSprop])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [20])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_optimizers_compare_to_reference(self, model_config, optimizer, micro_batch_size,
                                           nbatches, number_epochs):
    tnt_history, ref_history = train_tnt_and_reference_models(model_config, optimizer,
                                             micro_batch_size, nbatches, number_epochs)
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'], 1e-2)
    assert np.allclose(tnt_history.history[metric], ref_history.history[metric], 1e-2)

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.lenet5_model_generator, False),
                                            base_runner.ModelConfig(mnist.sequential_model_generator)])
  @pytest.mark.parametrize("nesterov", [False, True])
  @pytest.mark.parametrize("momentum", [0.9])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_sgd_momentum_compare_to_reference(self, model_config, nesterov, momentum,
                                             micro_batch_size, nbatches, number_epochs):
    optimizer = keras.optimizers.SGD
    optimizer_kwargs = {'learning_rate' : 0.01,
                        'momentum' : momentum,
                        'nesterov' : nesterov}
    tnt_history, ref_history = train_tnt_and_reference_models(model_config, optimizer,
                                             micro_batch_size, nbatches, number_epochs,
                                             optimizer_kwargs)
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'], 1e-4)
    assert np.allclose(tnt_history.history[metric], ref_history.history[metric], 1e-6)
