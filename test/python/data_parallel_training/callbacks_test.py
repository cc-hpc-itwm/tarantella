from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras

import pytest

@pytest.fixture(scope="class", params=[mnist.sequential_model_generator,
                                      ])
def model_runner(request):
  yield base_runner.generate_tnt_model_runner(request.param())

def train_val_dataset_generator():
  micro_batch_size = 64
  nbatches = 1

  comm_size = tnt.get_size()
  batch_size = micro_batch_size * comm_size
  nsamples = nbatches * batch_size

  (train_dataset, val_dataset) = util.load_dataset(mnist.load_mnist_dataset,
                                                   train_size = nsamples,
                                                   train_batch_size = batch_size,
                                                   test_size = 1000,
                                                   test_batch_size = batch_size)
  yield from (train_dataset, val_dataset)

class TestsDataParallelCallbacks:
  def test_model_initialization(self, model_runner):
    assert model_runner.model

  def test_history_default_callback(self, model_runner):
    number_epochs = 2
    (train_dataset, val_dataset) = train_val_dataset_generator()

    history = model_runner.model.fit(train_dataset,
                                     epochs=number_epochs,
                                     verbose=0,
                                     shuffle=False,
                                     validation_data=val_dataset)

    assert len(history.history['val_loss']) == number_epochs

  def test_early_stopping_callback(self, model_runner):
    number_epochs = 10
    (train_dataset, val_dataset) = train_val_dataset_generator()

    history = model_runner.model.fit(train_dataset,
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
