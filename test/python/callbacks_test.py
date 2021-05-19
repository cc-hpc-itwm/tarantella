from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import pytest
import shutil

@pytest.fixture(scope="function", params=[mnist.fc_model_generator,
                                          mnist.subclassed_model_generator])
def model_runners(request):
  tnt_model_runner = base_runner.generate_tnt_model_runner(request.param())
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield tnt_model_runner, reference_model_runner


@pytest.fixture(scope="function")
def setup_save_path(request):
  barrier = tnt.Barrier()
  barrier.synchronize()
  # save logs in a shared directory accessible to all ranks
  save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_callbacks")
  if tnt.is_master_rank():
    os.makedirs(save_dir, exist_ok=True)
  yield save_dir

  # clean up
  barrier.synchronize()
  if tnt.is_master_rank():
    shutil.rmtree(save_dir, ignore_errors=True)


def train_val_dataset_generator():
  micro_batch_size = 64
  nbatches = 1
  batch_size = micro_batch_size * tnt.get_size()
  nsamples = nbatches * batch_size

  return util.load_dataset(mnist.load_mnist_dataset,
                           train_size = nsamples,
                           train_batch_size = batch_size,
                           test_size = nsamples,
                           test_batch_size = batch_size)

class TestsDataParallelCallbacks:
  def train_tnt_and_ref_models_with_callbacks(self, callbacks, model_runners, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()
  
    tnt_model_runner, reference_model_runner = model_runners

    param_dict = { 'epochs' : number_epochs,
                   'verbose' : 0,
                   'shuffle' : False,
                   'callbacks' : callbacks }
    tnt_history = tnt_model_runner.model.fit(train_dataset,
                                             validation_data=val_dataset,
                                             **param_dict)
    ref_history = reference_model_runner.model.fit(ref_train_dataset,
                                                   validation_data=ref_val_dataset,
                                                   **param_dict)
    return (tnt_history, ref_history)
  
  @pytest.mark.parametrize("number_epochs", [5])
  def test_learning_rate_scheduler_callback(self, model_runners, number_epochs):
    callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule=(lambda epoch, lr: 0.1 * lr),
                                                          verbose=1)]
    tnt_history, reference_history = self.train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_runners, number_epochs)

    for key in reference_history.history.keys():
      assert all(np.isclose(tnt_history.history[key], reference_history.history[key], atol=1e-6))

  @pytest.mark.parametrize("number_epochs", [1])
  def test_tensorboard_callback(self, setup_save_path, model_runners, number_epochs):
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir = setup_save_path)]
    self.train_tnt_and_ref_models_with_callbacks(callbacks, model_runners, number_epochs)
    # FIXME: assert correct file exists
    assert True

  @pytest.mark.parametrize("number_epochs", [1])
  def test_model_checkpoint_callback(self, setup_save_path, model_runners, number_epochs):
    checkpoint_path = os.path.join(setup_save_path, "logs")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path)]
    self.train_tnt_and_ref_models_with_callbacks(callbacks, model_runners, number_epochs)
    # FIXME: assert correct file exists
    assert True

  @pytest.mark.parametrize("number_epochs", [1])
  def test_history_callback(self, model_runners, number_epochs):
    # history callback is added by default
    callbacks = []
    tnt_history, reference_history = self.train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_runners, number_epochs)

    for key in reference_history.history.keys():
      assert all(np.isclose(tnt_history.history[key], reference_history.history[key], atol=1e-6))

  @pytest.mark.parametrize("number_epochs", [10])
  def test_early_stopping_callback(self, model_runners, number_epochs):
    monitor_metric = 'val_loss'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor_metric,
                                                  min_delta=0.1,
                                                  patience=1)]
    tnt_history, reference_history = self.train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_runners, number_epochs)

    # Expect both models to run same number of epochs
    assert len(tnt_history.history[monitor_metric]) == len(reference_history.history[monitor_metric])

  @pytest.mark.parametrize("number_epochs", [2])
  def test_csv_logger_callback(self, setup_save_path, model_runners, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

    filename = os.path.join(setup_save_path, "training")
    tnt_model_runner, reference_model_runner = model_runners
    param_dict = { 'epochs' : number_epochs,
                   'verbose' : 0,
                   'shuffle' : False }

    tnt_filename = filename + '_tnt.csv'
    tnt_model_runner.model.fit(train_dataset,
                               validation_data=val_dataset,
                               callbacks = [tf.keras.callbacks.CSVLogger(tnt_filename)],
                               **param_dict)

    if tnt.is_master_rank():
      ref_filename = filename + '_ref.csv'
      reference_model_runner.model.fit(ref_train_dataset,
                                      validation_data=ref_val_dataset,
                                      callbacks = [tf.keras.callbacks.CSVLogger(ref_filename)],
                                      **param_dict)

      tnt_metrics = util.get_metric_values_from_file(tnt_filename)
      ref_metrics = util.get_metric_values_from_file(ref_filename)
      assert np.allclose(tnt_metrics, ref_metrics, atol = 1e-6)

  @pytest.mark.parametrize("number_epochs", [1])
  def test_terminate_callback(self, model_runners, number_epochs):
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    tnt_history, ref_history = self.train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_runners, number_epochs)

    for key in ref_history.history.keys():
      assert all(np.isclose(tnt_history.history[key], ref_history.history[key], atol=1e-6))

  #FIXME: Remove version tag when BaseLogger is fixed on Tensorflow
  @pytest.mark.max_tfversion('2.1')
  @pytest.mark.parametrize("number_epochs", [1])
  def test_base_logger_callback(self, model_runners, number_epochs):
    callbacks = [tf.keras.callbacks.BaseLogger()]
    tnt_history, ref_history = self.train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_runners, number_epochs, verbose=1)

    for key in ref_history.history.keys():
      assert all(np.isclose(tnt_history.history[key], ref_history.history[key], atol=1e-6))