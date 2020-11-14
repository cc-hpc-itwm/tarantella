import argparse
import tensorflow as tf
from tensorflow import keras

import tarantella as tnt
tnt.init()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-bs", "--batch_size", type=int, default=64)
  parser.add_argument("-e", "--number_epochs", type=int, default=1)
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
  parser.add_argument("-train", "--train_size", type=int, default=48000)
  parser.add_argument("-val", "--val_size", type=int, default=6400)
  parser.add_argument("-test", "--test_size", type=int, default=6400)
  args = parser.parse_args()
  return args

def mnist_as_np_arrays(training_samples, validation_samples, test_samples):
  mnist_train_size = 60000
  mnist_test_size = 10000
  assert(training_samples + validation_samples <= mnist_train_size)
  assert(test_samples <= mnist_test_size)

  # load given number of samples
  (x_train_all, y_train_all), (x_test_all, y_test_all) = \
        keras.datasets.mnist.load_data()
  x_train = x_train_all[:training_samples]
  y_train = y_train_all[:training_samples]
  x_val = x_train_all[training_samples:training_samples+validation_samples]
  y_val = y_train_all[training_samples:training_samples+validation_samples]
  x_test = x_test_all[:test_samples]
  y_test = y_test_all[:test_samples]

  # normalization and reshape
  x_train = x_train.reshape(training_samples,28,28,1).astype('float32') / 255.
  x_val = x_val.reshape(validation_samples,28,28,1).astype('float32') / 255.
  x_test = x_test.reshape(test_samples,28,28,1).astype('float32') / 255.
  y_train = y_train.astype('float32')
  y_val = y_val.astype('float32')
  y_test = y_test.astype('float32')

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def lenet5_model_generator():
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(20, 5, padding="same", activation='relu')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = keras.layers.Conv2D(50, 5, padding="same", activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(500, activation='relu')(x)
  outputs = keras.layers.Dense(10, activation='softmax')(x)
  return keras.Model(inputs=inputs, outputs=outputs)

args = parse_args()
              
# Create Tarantella model
model = tnt.Model(lenet5_model_generator())

# Compile Tarantella model (as with Keras)
model.compile(optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = [keras.metrics.SparseCategoricalAccuracy()])

# Load MNIST dataset (as with Keras)
shuffle_seed = 42
(x_train, y_train), (x_val, y_val), (x_test, y_test) = \
      mnist_as_np_arrays(args.train_size, args.val_size, args.test_size)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(len(x_train), shuffle_seed)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(args.batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(args.batch_size)

# Train Tarantella model (as with Keras)
model.fit(train_dataset,
          validation_data = val_dataset,
          epochs = args.number_epochs,
          verbose = 1)

# Evaluate Tarantella model (as with Keras)
model.evaluate(test_dataset, verbose = 1)
