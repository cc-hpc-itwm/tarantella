from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pytest

class TestsModelAPI:
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  def test_model_api_methods(self, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tnt.get_size()
    nsamples = nbatches * batch_size
    (number_epochs, lr) = mnist.get_hyperparams(keras.optimizers.Adam)
    (train_dataset, test_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 10000,
                                                      test_batch_size = batch_size)
    model_runner = base_runner.generate_tnt_model_runner(mnist.lenet5_model_generator())
    
    # Add loss and metric
    model_runner.model.add_loss(lambda: tf.reduce_mean(model_runner.model.layers[-2].kernel))
    model_runner.model.add_metric(model_runner.model.model.output, name='mean_output', aggregation='mean')

    model_runner.reset_weights()
    model_runner.train_model(train_dataset, number_epochs)
    results = model_runner.evaluate_model(test_dataset)
    util.check_accuracy_greater(results[1], 0.5)
    # Test model output shape
    assert model_runner.model.compute_output_shape((None, 28, 28, 1))[1] == 10

  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  def test_model_api_attributes(self, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tnt.get_size()
    nsamples = nbatches * batch_size
    (number_epochs, _) = mnist.get_hyperparams(keras.optimizers.Adam)
    (train_dataset, _) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 0,
                                                      test_batch_size = batch_size)
    model_runner = base_runner.generate_tnt_model_runner(mnist.sequential_model_generator())

    model_runner.reset_weights()
    model_runner.train_model(train_dataset, number_epochs)

    assert model_runner.model.layers[1] == model_runner.model.get_layer(index=1)
    
    assert len(model_runner.model.trainable_weights) == 6
    assert len(model_runner.model.non_trainable_weights) == 0
    assert len(model_runner.model.weights) == 6
    assert model_runner.model.dynamic == False
  
  @pytest.mark.parametrize("micro_batch_size", [64])
  @pytest.mark.parametrize("nbatches", [230])
  def test_model_api_metrics(self, micro_batch_size, nbatches):
    batch_size = micro_batch_size * tnt.get_size()
    nsamples = nbatches * batch_size
    (number_epochs, _) = mnist.get_hyperparams(keras.optimizers.Adam)
    (train_dataset, _) = util.load_dataset(mnist.load_mnist_dataset,
                                                      train_size = nsamples,
                                                      train_batch_size = batch_size,
                                                      test_size = 0,
                                                      test_batch_size = batch_size)
    model_runner = base_runner.generate_tnt_model_runner(mnist.sequential_model_generator())\

    model_runner.reset_weights()
    model_runner.train_model(train_dataset, number_epochs)

    model_runner.model.metrics[0].reset_states()
    model_runner.model.reset_metrics()
    assert model_runner.model.metrics[0].result().numpy() == 0.0
    assert len(model_runner.model.metrics_names) == 2
