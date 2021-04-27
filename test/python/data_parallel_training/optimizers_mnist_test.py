from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow.keras as keras
import numpy as np

import logging
import pytest

def model_runners(model_generator):
  tnt_model_runner = base_runner.generate_tnt_model_runner(model_generator())
  reference_model_runner = base_runner.TrainingRunner(model_generator())
  return tnt_model_runner, reference_model_runner

def get_compiled_models(model_generator, optimizer, **optimizer_kwargs):
  tnt_model_runner, ref_model_runner = model_runners(model_generator)
  tnt_model_runner.compile_model(optimizer(**optimizer_kwargs))
  ref_model_runner.compile_model(optimizer(**optimizer_kwargs))

  return tnt_model_runner, ref_model_runner


class TestsDataParallelOptimizers:
  @pytest.mark.parametrize("model_generator", [mnist.lenet5_model_generator,
                                               mnist.subclassed_model_generator])
  @pytest.mark.parametrize("optimizer", [keras.optimizers.Adadelta,
                                         keras.optimizers.Adagrad,
                                         keras.optimizers.Adam,
                                         keras.optimizers.Adamax,
                                         #keras.optimizers.Nadam,
                                         keras.optimizers.RMSprop,
                                         keras.optimizers.SGD,
                                        ])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [15])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_optimizers_compare_to_reference(self, model_generator, optimizer, micro_batch_size,
                                           nbatches, number_epochs):
    (train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                        micro_batch_size = micro_batch_size)
    (ref_train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                            micro_batch_size = micro_batch_size)
    tnt_model_runner, ref_model_runner = get_compiled_models(model_generator, optimizer)

    tnt_history = tnt_model_runner.train_model(train_dataset, number_epochs)
    ref_history = ref_model_runner.train_model(ref_train_dataset, number_epochs)

    rank = tnt.get_rank()
    metric = 'sparse_categorical_accuracy'
    logging.getLogger().info(f"[Rank {rank}] Tarantella (loss, accuracy) = "
                             f"({tnt_history.history['loss']}, {tnt_history.history[metric]})")
    logging.getLogger().info(f"[Rank {rank}] Reference (loss, accuracy) = "
                             f"({ref_history.history['loss']}, {ref_history.history[metric]})")
    # compare histories (note that RMSProp performs worse than the other optimizers)
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'],
                       atol=1e-2 if isinstance(optimizer(),  keras.optimizers.RMSprop) else 1e-4)
    assert np.allclose(tnt_history.history[metric], ref_history.history[metric],
                       atol=1e-2 if isinstance(optimizer(),  keras.optimizers.RMSprop) else 1e-6)

  @pytest.mark.parametrize("model_generator", [mnist.lenet5_model_generator,
                                               mnist.sequential_model_generator])
  @pytest.mark.parametrize("nesterov", [False, True])
  @pytest.mark.parametrize("momentum", [0.9])
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("nbatches", [10])
  @pytest.mark.parametrize("number_epochs", [2])
  def test_sgd_momentum_compare_to_reference(self, model_generator, nesterov, momentum,
                                             micro_batch_size, nbatches, number_epochs):
    (train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                        micro_batch_size = micro_batch_size)
    (ref_train_dataset, _) = util.train_test_mnist_datasets(nbatches = nbatches,
                                                            micro_batch_size = micro_batch_size)
    optimizer = keras.optimizers.SGD
    optimizer_kwargs = {'learning_rate' : 0.01,
                        'momentum' : momentum,
                        'nesterov' : nesterov}
    tnt_model_runner, ref_model_runner = get_compiled_models(model_generator, optimizer,
                                                             **optimizer_kwargs)
    tnt_history = tnt_model_runner.train_model(train_dataset, number_epochs)
    ref_history = ref_model_runner.train_model(ref_train_dataset, number_epochs)

    rank = tnt.get_rank()
    metric = 'sparse_categorical_accuracy'
    logging.getLogger().info(f"[Rank {rank}] Tarantella (loss, accuracy) = "
                             f"({tnt_history.history['loss']}, {tnt_history.history[metric]})")
    logging.getLogger().info(f"[Rank {rank}] Reference (loss, accuracy) = "
                             f"({ref_history.history['loss']}, {ref_history.history[metric]})")
    assert np.allclose(tnt_history.history['loss'], ref_history.history['loss'], 1e-4)
    assert np.allclose(tnt_history.history[metric], ref_history.history[metric], 1e-6)
