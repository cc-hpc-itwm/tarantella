import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import logging

# Optimizer Hyperparameters
# Dictionary: Optimizer Name: (number_epochs, learning_rate)
hyperparams_cifar = {'Adadelta': (12, 1),
                    'Adagrad':   (20, 0.05),
                    'Adam':      (5,  0.001),
                    'Adamax':    (10, 0.001),
                    'Nadam':     (10, 0.0001),
                    'RMSprop':   (10, 0.001),
                    'SGD':       (20, 0.01)}

def get_hyperparams(optimizer):
  opt = optimizer.__name__
  return hyperparams_cifar.get(opt)

# Load CIFAR-10 dataset
def load_cifar_dataset(training_samples, validation_samples, test_samples):
  cifar_train_size = 60000
  cifar_test_size = 10000
  assert(training_samples + validation_samples <= cifar_train_size)
  assert(test_samples <= cifar_test_size)

  # load given number of samples
  (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.cifar10.load_data()
  x_train = x_train_all[:training_samples]
  y_train = y_train_all[:training_samples]
  x_val = x_train_all[training_samples:training_samples+validation_samples]
  y_val = y_train_all[training_samples:training_samples+validation_samples]
  x_test = x_test_all[:test_samples]
  y_test = y_test_all[:test_samples]

  # Preprocess the data (these are Numpy arrays)
  x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
  x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255
  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def alexnet_model_generator():
  inputs = keras.Input(shape=(32,32,3,), name='input')
  x = layers.Conv2D(96, 3, strides=(4, 4), activation='relu')(inputs)
  x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = layers.Conv2D(256, 5, padding='same', activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = layers.Conv2D(384, 3, padding='same', activation='relu')(x)
  x = layers.Conv2D(384, 3, padding='same', activation='relu')(x)
  x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(4096, activation='relu')(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(4096, activation='relu')(x)
  x = layers.Dropout(0.4)(x)
  outputs = layers.Dense(10, activation='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  
  logging.getLogger().info("Initialized AlexNet model")
  return model
