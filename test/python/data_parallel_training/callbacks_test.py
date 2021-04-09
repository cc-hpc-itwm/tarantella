from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras

import numpy as np

import pytest
import logging

@pytest.fixture(scope="class", params=[mnist.sequential_model_generator,
                                      ])
def model_runners(request):
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner

def train_val_dataset_generator():
  micro_batch_size = 64
  nbatches = 1

  comm_size = tnt.get_size()
  batch_size = micro_batch_size * comm_size
  nsamples = nbatches * batch_size

  (train_dataset, val_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                   train_size = nsamples,
                                                   train_batch_size = batch_size,
                                                   test_size = 100,
                                                   test_batch_size = batch_size)
  yield from (train_dataset, val_dataset)

class TestsDataParallelCallbacks:
  @pytest.mark.parametrize("number_epochs", [2])
  def test_history_default_callback(self, model_runners, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

    tnt_model_runner, reference_model_runner = model_runners

    tnt_history = tnt_model_runner.model.fit(train_dataset,
                                             epochs=number_epochs,
                                             verbose=0,
                                             shuffle=False,
                                             validation_data=val_dataset)
    reference_history = reference_model_runner.model.fit(ref_train_dataset,
                                                         epochs=number_epochs,
                                                         verbose=0,
                                                         shuffle=False,
                                                         validation_data=ref_val_dataset)                                        

    for key in tnt_history.history.keys():
      assert all(np.isclose(tnt_history.history[key], reference_history.history[key], atol=3e-1))

    assert len(history.history['val_loss']) == number_epochs

  @pytest.mark.parametrize("number_epochs", [10])
  def test_early_stopping_callback(self, model_runners, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()

    tnt_model_runner, _ = model_runners
    history = tnt_model_runner.model.fit(train_dataset,
                                     epochs=number_epochs,
                                     verbose=0,
                                     shuffle=False,
                                     validation_data=val_dataset,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                 min_delta=0.1,
                                                                                 patience=1,
                                                                                 verbose=1)],
                                     )

    assert len(history.history['val_loss']) < number_epochs
