import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import logging

# Optimizer Hyperparameters
# Dictionary: Optimizer Name: (number_epochs, learning_rate)
hyperparams_mnist = {'Adadelta': (1, 1),
                    'Adagrad':   (3, 0.01),
                    'Adam':      (1, 0.001),
                    'Adamax':    (2, 0.001),
                    'Nadam':     (1, 0.002),
                    'RMSprop':   (1, 0.001),
                    'SGD':       (8, 0.01)}

def get_hyperparams(optimizer):
  opt = optimizer.__name__
  return hyperparams_mnist.get(opt)

# Load MNIST dataset
def load_mnist_dataset(training_samples, validation_samples, test_samples):
  mnist_train_size = 60000
  mnist_test_size = 10000
  assert(training_samples + validation_samples <= mnist_train_size)
  assert(test_samples <= mnist_test_size)

  # load given number of samples
  (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.mnist.load_data()
  x_train = x_train_all[:training_samples]
  y_train = y_train_all[:training_samples]
  x_val = x_train_all[training_samples:training_samples+validation_samples]
  y_val = y_train_all[training_samples:training_samples+validation_samples]
  x_test = x_test_all[:test_samples]
  y_test = y_test_all[:test_samples]

  # normalization and reshape
  x_train = x_train.reshape(training_samples, 28, 28, 1).astype('float32') / 255.
  x_val = x_val.reshape(validation_samples, 28, 28, 1).astype('float32') / 255.
  x_test = x_test.reshape(test_samples, 28, 28, 1).astype('float32') / 255.
  y_train = y_train.astype('float32')
  y_val = y_val.astype('float32')
  y_test = y_test.astype('float32')

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def fc_model_generator():
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = layers.Flatten()(inputs)
  x = layers.Dense(200, activation='relu', name='FC1')(x)
  x = layers.Dense(200, activation='relu', name='FC2')(x)
  outputs = layers.Dense(10, activation='softmax', name='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  logging.getLogger().info("Initialized FC model")
  return model

def lenet5_model_generator():
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = layers.Conv2D(20, 5, padding="same", activation='relu')(inputs)
  x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = layers.Conv2D(50, 5, padding="same", activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(500, activation='relu')(x)
  outputs = layers.Dense(10, activation='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  logging.getLogger().info("Initialized LeNet5 model")
  return model

def sequential_model_generator():
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28,28,1,)))
  model.add(layers.Dense(200, activation='relu', name='FC1'))
  model.add(layers.Dense(200, activation='relu', name='FC2'))
  model.add(layers.Dense(10, activation='softmax', name='softmax'))

  logging.getLogger().info("Initialized Sequential model")
  return model

def alexnet_model_generator():
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(inputs)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  outputs = layers.Dense(10, activation='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)

  logging.getLogger().info("Initialized AlexNet model")
  return model

class SubclassedModel(tf.keras.Model):
  def __init__(self):
    super(SubclassedModel, self).__init__()
    self.flatten = keras.layers.Flatten(input_shape=(28,28,1,))
    self.dense = keras.layers.Dense(200, activation='relu', name='FC')
    self.classifier = keras.layers.Dense(10, activation='softmax', name='softmax')
    logging.getLogger().info("Initialized SubclassedModel")

  def call(self, inputs):
    x = self.flatten(inputs)
    x = self.dense(x)
    return self.classifier(x)

def subclassed_model_generator():
  model = SubclassedModel()
  model.build((None,28,28,1))
  return model