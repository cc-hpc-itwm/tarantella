from models import mnist_models as mnist
from tensorflow import keras
import utilities as util
import tarantella as tnt
import numpy as np

import logging
import pytest

def custom_train(model, train_dataset, number_epochs):
  for epoch in range(number_epochs):
    # Iterate over the batches of the dataset.
    for step, (train_batch) in enumerate(train_dataset):
      train_metrics = model.train_step(train_batch)
      # Log every 200 batches.
      if step % 200 == 0:
        print(f"Training loss (for one batch) at step {step}: {train_metrics['loss']}")

    # Display metrics at the end of each epoch.
    print(f"Training acc over epoch: {train_metrics['mae']}")
    # Reset metrics at end of each epoch
    model.reset_metrics()

def custom_evaluate(model, test_dataset):
  # Run a validation loop at the end of each epoch.
  for test_batch in test_dataset:
    test_metrics = model.test_step(test_batch)

  val_loss = test_metrics['loss']
  val_acc = test_metrics['mae']
  print(f"Validation acc: {val_acc}")
  model.reset_metrics()
  return val_loss, val_acc


@pytest.mark.min_tfversion('2.2')
class TestsStepTrainingAccuracy:
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [3])
  @pytest.mark.parametrize("nbatches", [20])
  @pytest.mark.parametrize("test_nbatches", [10])
  def test_compare_custom_training_loop(self, micro_batch_size,
                                        number_epochs, nbatches, test_nbatches):
    (keras_train_dataset, keras_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                   micro_batch_size, shuffle = False)
    (tnt_train_dataset, tnt_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size, shuffle = False)
    shuffle_seed = 42
    util.set_tf_random_seed()

    # Instantiate, compile and train keras model
    keras_model = mnist.sequential_model_generator()
    keras_model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    custom_train(keras_model, keras_train_dataset, number_epochs)

    # Distribute train and test dataset among ranks
    distributed_x_train = tnt.data.Dataset(tnt_train_dataset, shuffle_seed)
    distributed_tnt_train_dataset = distributed_x_train.distribute_dataset_across_ranks(is_training = True)
    distributed_x_test = tnt.data.Dataset(tnt_test_dataset, shuffle_seed)
    distributed_tnt_test_dataset = distributed_x_test.distribute_dataset_across_ranks(is_training = False)

    # Instantiate, compile and train tarantella model
    tnt_model = tnt.Model(mnist.sequential_model_generator())
    tnt_model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    custom_train(tnt_model, distributed_tnt_train_dataset, number_epochs)

    # Run both models on test dataset
    keras_loss_accuracy = custom_evaluate(keras_model, keras_test_dataset)
    tnt_loss_accuracy = custom_evaluate(tnt_model, distributed_tnt_test_dataset)

    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Keras [loss, accuracy] = {keras_loss_accuracy}")
    assert np.isclose(tnt_loss_accuracy[0], keras_loss_accuracy[0], atol=1e-2) # losses might not be identical
    assert np.isclose(tnt_loss_accuracy[1], keras_loss_accuracy[1], atol=1e-6)
