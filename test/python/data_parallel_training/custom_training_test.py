from models import mnist_models as mnist
import tensorflow as tf
from tensorflow import keras
import utilities as util
import tarantella as tnt

import numpy as np

import logging
import pytest

def custom_train(model, train_dataset, loss_fn, optimizer, metric, number_epochs):
  # Train step function
  @tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metric.update_state(y, logits)
    return loss_value

  train_loss_acc = []
  for epoch in range(number_epochs):
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      loss_value = train_step(x_batch_train, y_batch_train)

    # Save (loss, accuracy) tuple
    train_loss_acc.append((loss_value.numpy(), metric.result().numpy()))

    # Reset training metrics at the end of each epoch
    metric.reset_states()
  return train_loss_acc

@pytest.mark.xfail(reason="Distributed metrics not yet implemented")
class TestsCustomTrainingAccuracy:
  @pytest.mark.min_tfversion('2.2')
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [1])
  @pytest.mark.parametrize("nbatches", [20])
  def test_compare_custom_training_loop(self, micro_batch_size, number_epochs, nbatches):
    (keras_train_dataset, _) = util.train_test_mnist_datasets(nbatches, 0, micro_batch_size)
    (tnt_train_dataset, _) = util.train_test_mnist_datasets(nbatches, 0, micro_batch_size)

    # Instantiate Keras loss , optimizer, meric and train model
    keras_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_keras_metric = keras.metrics.SparseCategoricalAccuracy()

    keras_model = mnist.sequential_model_generator()
    keras_history = custom_train(keras_model, keras_train_dataset,
                                 loss_fn, keras_optimizer, train_keras_metric, number_epochs)

    # Distribute train dataset among ranks
    distributed_x_train = tnt.data.Dataset(dataset = tnt_train_dataset)
    distributed_tnt_train_dataset = distributed_x_train.distribute_dataset_across_ranks(is_training = True)

    # Instantiate tnt model, optimizer and train model
    tnt_model = tnt.Model(mnist.sequential_model_generator())
    tnt_optimizer = tnt.Optimizer(keras.optimizers.Adam(learning_rate=0.001))
    train_tnt_metric = keras.metrics.SparseCategoricalAccuracy()
    tnt_history = custom_train(tnt_model, distributed_tnt_train_dataset,
                               loss_fn, tnt_optimizer, train_tnt_metric, number_epochs)

    rank = tnt.get_rank()
    for tnt_loss_accuracy, keras_loss_accuracy in zip(tnt_history, keras_history):
      logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
      logging.getLogger().info(f"[Rank {rank}] Keras [loss, accuracy] = {keras_loss_accuracy}")
      assert np.isclose(tnt_loss_accuracy[0], keras_loss_accuracy[0], atol=1e-2) # losses might not be identical
      assert np.isclose(tnt_loss_accuracy[1], keras_loss_accuracy[1], atol=1e-6)
