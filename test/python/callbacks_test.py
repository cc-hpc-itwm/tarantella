# CustomLearningRateScheduler:
#   Copyright (C) 2021 keras.io <https://www.tensorflow.org/guide/keras/custom_callback#learning_rate_scheduling>
#   Modifications Copyright (C) 2022 Fraunhofer ITWM <http://www.itwm.fraunhofer.de/>

from models import mnist_models as mnist
from tarantella.strategy.parallel_strategy import ParallelStrategy
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np

import os
import pytest
import shutil

@pytest.fixture(scope="function")
def setup_save_path(request):
  barrier = tnt.Barrier()
  barrier.execute()
  # save logs in a shared directory accessible to all ranks
  save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_callbacks")
  if tnt.is_master_rank():
    os.makedirs(save_dir, exist_ok=True)
  yield save_dir

  # clean up
  barrier.execute()
  if tnt.is_master_rank():
    shutil.rmtree(save_dir, ignore_errors=True)


def gen_model_runners(model_config: base_runner.ModelConfig):
  tnt_model_runner = base_runner.generate_tnt_model_runner(model_config)
  reference_model_runner = base_runner.TrainingRunner(model_config.model_generator())
  return tnt_model_runner, reference_model_runner

def train_val_dataset_generator():
  micro_batch_size = 64
  nbatches = 1
  batch_size = micro_batch_size * tnt.get_size()
  nsamples = nbatches * batch_size
  train_dataset, val_dataset, _ = util.load_dataset(mnist.load_mnist_dataset,
                                                    train_size = nsamples,
                                                    train_batch_size = batch_size,
                                                    val_size = nsamples,
                                                    val_batch_size = batch_size)
  return train_dataset, val_dataset

class CustomLearningRateScheduler(keras.callbacks.Callback):
  # Learning rate scheduler to update the learning rate according to a schedule.
  def __init__(self, model = None):
    super().__init__()
    if model is not None:
      self.model = model
    self.step_schedule = (2, 4)
    self.step_size = 10
    
  def lr_schedule(self, epoch, lr):
    if epoch in self.step_schedule:
      return lr/self.step_size
    return lr

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, "lr"):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.lr_schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


def train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs):
  (train_dataset, val_dataset) = train_val_dataset_generator()
  (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

  tnt_model_runner, reference_model_runner = gen_model_runners(model_config)

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

def assert_identical_tnt_and_ref_history(tnt_history, ref_history):
  result = [True]
  if tnt.is_master_rank():
    for key in ref_history.history.keys():
      result += [all(np.isclose(tnt_history.history[key], ref_history.history[key], atol=1e-6))]
    result = [all(result)]
  util.assert_on_all_ranks(result)

@pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator),
                                          base_runner.ModelConfig(mnist.subclassed_model_generator),
                                          pytest.param(base_runner.ModelConfig(mnist.fc_model_generator,
                                                                               tnt.ParallelStrategy.PIPELINING),
                                                       marks=pytest.mark.skipif(tnt.get_size() != 1,
                                                                                reason="Cannot run multi-rank, model has only one partition")),
                                          ])
class TestTarantellaCallbacks:
  @pytest.mark.parametrize("number_epochs", [5])
  def test_learning_rate_scheduler_callback(self, model_config, number_epochs):
    callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule=(lambda epoch, lr: 0.1 * lr),
                                                          verbose=1)]
    tnt_history, reference_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)
    assert_identical_tnt_and_ref_history(tnt_history, reference_history)

  
  @pytest.mark.parametrize("number_epochs", [5])
  def test_custom_callback(self, model_config, number_epochs):
    callbacks = [CustomLearningRateScheduler()]
    tnt_history, reference_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)
    assert_identical_tnt_and_ref_history(tnt_history, reference_history)


  @pytest.mark.parametrize("number_epochs", [5])
  def test_lambda_callback(self, model_config, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

    tnt_model_runner, reference_model_runner = gen_model_runners(model_config)

    lr_callback = CustomLearningRateScheduler(reference_model_runner.model)
    lr_lambda_callback = LambdaCallback(
      on_epoch_begin=lambda epoch,logs: lr_callback.on_epoch_begin(epoch, logs))
    
    lr_tnt_callback = CustomLearningRateScheduler(tnt_model_runner.model.model)
    lr_tnt_lambda_callback = LambdaCallback(
      on_epoch_begin=lambda epoch,logs: lr_tnt_callback.on_epoch_begin(epoch, logs))
    
    callbacks = [lr_lambda_callback]
    tnt_callbacks = [tnt.keras.callbacks.Callback(lr_tnt_lambda_callback, aggregate_logs=False, run_on_all_ranks=True)]
    
    tnt_history = tnt_model_runner.model.fit(train_dataset,
                                             validation_data=val_dataset,
                                             epochs=number_epochs,
                                             verbose=0,
                                             shuffle=False,
                                             callbacks=tnt_callbacks)
    reference_history = reference_model_runner.model.fit(ref_train_dataset,
                                                   validation_data=ref_val_dataset,
                                                   epochs=number_epochs,
                                                   verbose=0,
                                                   shuffle=False,
                                                   callbacks=callbacks)
    assert_identical_tnt_and_ref_history(tnt_history, reference_history)


  @pytest.mark.parametrize("number_epochs", [1])
  def test_tensorboard_callback(self, setup_save_path, model_config, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    tnt_model_runner, _ = gen_model_runners(model_config)

    tnt_model_runner.model.fit(train_dataset, validation_data=val_dataset,
                               epochs = number_epochs,
                               callbacks = [tf.keras.callbacks.TensorBoard(log_dir = setup_save_path)])
    result = [True]
    if tnt.is_master_rank():
      result = [os.path.isdir(os.path.join(setup_save_path, "train")),
                os.path.isdir(os.path.join(setup_save_path, "validation"))]
      result = [all(result)]
    util.assert_on_all_ranks(result)

  @pytest.mark.parametrize("number_epochs", [1])
  def test_history_callback(self, model_config, number_epochs):
    # history callback is added by default
    callbacks = []
    tnt_history, reference_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)
    assert_identical_tnt_and_ref_history(tnt_history, reference_history)


  @pytest.mark.parametrize("number_epochs", [10])
  def test_early_stopping_callback(self, model_config, number_epochs):
    monitor_metric = 'val_loss'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor_metric,
                                                  min_delta=0.1,
                                                  patience=1)]
    tnt_history, reference_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)

    # Expect both models to run same number of epochs
    result = [True]
    if tnt.is_master_rank():
      result = (len(tnt_history.history[monitor_metric]) == len(reference_history.history[monitor_metric]))
    util.assert_on_all_ranks(result)

  @pytest.mark.parametrize("number_epochs", [2])
  def test_csv_logger_callback(self, setup_save_path, model_config, number_epochs):
    (train_dataset, val_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

    filename = os.path.join(setup_save_path, "training")
    tnt_model_runner, reference_model_runner = gen_model_runners(model_config)
    param_dict = { 'epochs' : number_epochs,
                   'verbose' : 0,
                   'shuffle' : False }

    tnt_filename = filename + '_tnt.csv'
    tnt_model_runner.model.fit(train_dataset,
                               validation_data=val_dataset,
                               callbacks = [tf.keras.callbacks.CSVLogger(tnt_filename)],
                               **param_dict)

    result = [True]
    if tnt.is_master_rank():
      ref_filename = filename + '_ref.csv'
      reference_model_runner.model.fit(ref_train_dataset,
                                      validation_data=ref_val_dataset,
                                      callbacks = [tf.keras.callbacks.CSVLogger(ref_filename)],
                                      **param_dict)

      tnt_metrics = util.get_metric_values_from_file(tnt_filename)
      ref_metrics = util.get_metric_values_from_file(ref_filename)
      result = np.allclose(tnt_metrics, ref_metrics, atol = 1e-6)
    util.assert_on_all_ranks(result)


  # FIXME: This does not seem to even trigger a `NaN`
  @pytest.mark.parametrize("number_epochs", [1])
  def test_terminate_callback(self, model_config, number_epochs):
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    tnt_history, ref_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)
    assert_identical_tnt_and_ref_history(tnt_history, ref_history)


  @pytest.mark.parametrize("number_epochs", [1])
  def test_base_logger_callback(self, model_config, number_epochs):
    callbacks = [tf.keras.callbacks.BaseLogger()]
    with pytest.raises(ValueError):
      tnt_history, ref_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)


  @pytest.mark.parametrize("number_epochs", [3])
  def test_reduce_lr_on_plateau_callback(self, model_config, number_epochs):
    monitor_metric = 'val_loss'
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric,
                                                      factor=0.01,
                                                      min_delta=0.01,
                                                      patience=1)]
    tnt_history, reference_history = train_tnt_and_ref_models_with_callbacks(
                                       callbacks, model_config, number_epochs)
    assert_identical_tnt_and_ref_history(tnt_history, reference_history)


  @pytest.mark.min_tfversion('2.4')
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("use_explicit_progbarlogger", [True, False])
  @pytest.mark.parametrize("verbose", [0, 1, 2])
  @pytest.mark.parametrize("exec_type", ['fit_with_validation', 'fit_without_validation'])
  def test_progbar_logger_callback_train(self, model_config, number_epochs,
                                             use_explicit_progbarlogger, verbose, exec_type, capsys):
    (train_dataset, test_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_test_dataset) = train_val_dataset_generator()
    if exec_type == 'fit_without_validation':
      test_dataset = None
      ref_test_dataset = None

    tnt_callbacks = [ tf.keras.callbacks.ProgbarLogger(count_mode = 'steps') ] if use_explicit_progbarlogger else []
    ref_callbacks = [ tf.keras.callbacks.ProgbarLogger(count_mode = 'steps') ] if use_explicit_progbarlogger else []

    tnt_model_runner, ref_model_runner = gen_model_runners(model_config)

    tnt_model_runner.model.fit(train_dataset, validation_data = test_dataset,
                                epochs = number_epochs, callbacks = tnt_callbacks,
                                verbose = verbose)
    tnt_captured = capsys.readouterr()
    tnt_metrics = util.get_metrics_from_stdout(tnt_captured.out, tnt_model_runner.model.metrics_names)

    ref_model_runner.model.fit(ref_train_dataset, validation_data = ref_test_dataset,
                                epochs = number_epochs, callbacks = ref_callbacks,
                                verbose = verbose)
    ref_captured = capsys.readouterr()
    ref_metrics = util.get_metrics_from_stdout(ref_captured.out, ref_model_runner.model.metrics_names)

    if tnt.is_master_rank():
      result = all(np.isclose(tnt_metrics, ref_metrics, atol=1e-6))
    else:
      result = all([tnt_captured.out == "", tnt_captured.err == ""])
    util.assert_on_all_ranks(result)

  @pytest.mark.min_tfversion('2.4')
  @pytest.mark.parametrize("number_epochs", [2])
  @pytest.mark.parametrize("use_explicit_progbarlogger", [True, False])
  @pytest.mark.parametrize("verbose", [2])  # FIXME: verbose = 1 does not issue the same values as the reference model
                                            # (becuse it processes micro-batches instead of batches)
  @pytest.mark.parametrize("exec_type", ['evaluate', 'predict'])
  def test_progbar_logger_callback_inference(self, model_config, number_epochs,
                                             use_explicit_progbarlogger, verbose, exec_type, capsys):
    (train_dataset, test_dataset) = train_val_dataset_generator()
    (ref_train_dataset, ref_test_dataset) = train_val_dataset_generator()

    tnt_callbacks = [ tf.keras.callbacks.ProgbarLogger(count_mode = 'steps') ] if use_explicit_progbarlogger else []
    ref_callbacks = [ tf.keras.callbacks.ProgbarLogger(count_mode = 'steps') ] if use_explicit_progbarlogger else []

    tnt_model_runner, ref_model_runner = gen_model_runners(model_config)

    if exec_type == 'evaluate':
      tnt_model_runner.model.evaluate(test_dataset, callbacks = tnt_callbacks, verbose = verbose)
    elif exec_type == 'predict':
      tnt_model_runner.model.predict(test_dataset, callbacks = tnt_callbacks, verbose = verbose)
    tnt_captured = capsys.readouterr()
    tnt_metrics = util.get_metrics_from_stdout(tnt_captured.out, tnt_model_runner.model.metrics_names)

    if exec_type == 'evaluate':
      ref_model_runner.model.evaluate(ref_test_dataset, callbacks = ref_callbacks, verbose = verbose)
    elif exec_type == 'predict':
      ref_model_runner.model.predict(ref_test_dataset, callbacks = ref_callbacks, verbose = verbose)
    ref_captured = capsys.readouterr()
    ref_metrics = util.get_metrics_from_stdout(ref_captured.out, ref_model_runner.model.metrics_names)

    if tnt.is_master_rank():
      result = all(np.isclose(tnt_metrics, ref_metrics, atol=1e-6))
    else:
      result = all([tnt_captured.out == "", tnt_captured.err == ""])
    util.assert_on_all_ranks(result)


class TestModelCheckpointCallback:
  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator),
                                            base_runner.ModelConfig(mnist.subclassed_model_generator)])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_model_checkpoint_data_par(self, setup_save_path, model_config, number_epochs):
    checkpoint_path = os.path.join(setup_save_path, "logs")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path)]
    train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs)
    # FIXME: assert correct file exists
    assert True

  @pytest.mark.parametrize("model_config", [base_runner.ModelConfig(mnist.fc_model_generator, tnt.ParallelStrategy.ALL)])
  @pytest.mark.parametrize("number_epochs", [1])
  def test_model_checkpoint_pipelining(self, setup_save_path, model_config, number_epochs):
    checkpoint_path = os.path.join(setup_save_path, "logs")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path)]
    with pytest.raises(ValueError):
      train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs)

