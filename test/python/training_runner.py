import tensorflow as tf
from tensorflow import keras

import tarantella as tnt
import utilities as util

def generate_tnt_model_runner(model):
  model_data_par = tnt.Model(model)
  runner = TrainingRunner(model_data_par)
  return runner

# Wrap tarantella model creation and compiling, as they should be executed only once
class TrainingRunner:
  def __init__(self, model):
    self.learning_rate = 0.001
    self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    self.loss = keras.losses.SparseCategoricalCrossentropy()
    self.metric = keras.metrics.SparseCategoricalAccuracy()
    self.model = model

    self.compile_model(self.optimizer)
    self.initial_weights = model.get_weights()

  def compile_model(self, optimizer):
    util.set_tf_random_seed()
    kwargs = {}
    if tf.__version__.startswith('2.0') or \
       tf.__version__.startswith('2.1'):
      kwargs['experimental_run_tf_function'] = False

    self.model.compile(optimizer=optimizer,
                       loss=self.loss,
                       metrics=[self.metric],
                       **kwargs)  # required for `keras` models

  def train_model(self, train_dataset, number_epochs):
    return self.model.fit(train_dataset,
                          epochs = number_epochs,
                          verbose = 0,
                          shuffle = False)  # required for `keras` models

  def get_weights(self):
    return self.model.get_weights()

  def reset_weights(self):
    self.model.set_weights(self.initial_weights)

  def evaluate_model(self, val_dataset):
    #return_dict to be added here (support only from tf 2.2)
    results = self.model.evaluate(val_dataset, verbose=0)
    return results

