import argparse
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.utils import keras_utils as utils

import tarantella

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-bs", "--batch_size", type=int, default=64)
  parser.add_argument("-micro", "--num_micro_batches", type=int, default=2)
  parser.add_argument("-e", "--number_epochs", type=int, default=1)
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
  parser.add_argument("-train", "--train_size", type=int, default=50000)
  parser.add_argument("-val", "--val_size", type=int, default=10000)
  parser.add_argument("-test", "--test_size", type=int, default=10000)
  parser.add_argument("-ngpus", "--ngpus_per_node", type=int, default=0)
  parser.add_argument("-v", "--verbose", type=int, default=0)
  args = parser.parse_args()
  return args

def mnist_as_np_arrays(training_samples, validation_samples, test_samples):
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

def create_dataset_from_arrays(samples, labels, batch_size):
  assert(len(samples) == len(labels))
  ds = tf.data.Dataset.from_tensor_slices((samples, labels))
  return ds.batch(batch_size)

args = parse_args()

tarantella.init(args.ngpus_per_node)
rank = tarantella.get_rank()
comm_size = tarantella.get_size()

batch_size = args.batch_size
micro_batch_size = args.batch_size // comm_size
train_size = args.train_size
val_size = args.val_size
test_size = args.test_size
shuffle_seed = 12

# MODEL
# Reference model
tf.random.set_seed(42)
inputs = keras.Input(shape=(28,28,1,), name='input')
x = layers.Flatten()(inputs)
x = layers.Dense(200, activation='relu', name='FC')(x)
x = layers.Dense(200, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax', name='softmax')(x)
reference_model = keras.Model(inputs=inputs, outputs=outputs)

# Building the graph
# Specify the training configuration (optimizer, loss, metrics) & compile
opt = keras.optimizers.SGD(learning_rate=args.learning_rate)
reference_model.compile(optimizer=opt,
                        loss=keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[keras.metrics.SparseCategoricalAccuracy()],
                        experimental_run_tf_function=False)
              
# Tarantella model
tf.random.set_seed(42)
inputs = keras.Input(shape=(28,28,1,), name='input')
x = layers.Flatten()(inputs)
x = layers.Dense(200, activation='relu', name='FC')(x)
x = layers.Dense(200, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax', name='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model = tarantella.model.TarantellaModel(model)

# Building the graph
# Specify the training configuration (optimizer, loss, metrics) & compile
opt = keras.optimizers.SGD(learning_rate=args.learning_rate)
model.compile(optimizer=opt,
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()],
              experimental_run_tf_function=False)

if rank == 0:
  model.summary()

# DATA LOADING & PRE-PROCESSING
# Load MNIST dataset
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_as_np_arrays(train_size, val_size, test_size)
reference_train_dataset = create_dataset_from_arrays(x_train, y_train, batch_size)
reference_val_dataset = create_dataset_from_arrays(x_val, y_val, micro_batch_size)
reference_test_dataset = create_dataset_from_arrays(x_test, y_test, micro_batch_size)

train_dataset = create_dataset_from_arrays(x_train, y_train, micro_batch_size)
val_dataset = create_dataset_from_arrays(x_val, y_val, micro_batch_size)
test_dataset = create_dataset_from_arrays(x_test, y_test, micro_batch_size)

reference_train_dataset = reference_train_dataset.shuffle(len(x_train), shuffle_seed)
train_dataset = train_dataset.shard(comm_size, rank).shuffle(len(x_train), shuffle_seed)


# TRAINING
# Reference model
history = reference_model.fit(reference_train_dataset,
                              epochs = args.number_epochs,
                              shuffle = False,
                              verbose= 1 if rank == 0 else 0,
                              validation_data=reference_val_dataset)
reference_loss_accuracy = reference_model.evaluate(reference_test_dataset,
                                                   verbose=0)

# Tarantella model
# Start the training w/ logging
log_file = "logs/profiler/gpi-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/rank" + str(rank)
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

runtime_callback = utils.RuntimeProfiler(batch_size = batch_size, logging_freq= 10, print_freq= 30)
history = model.fit(train_dataset,
                    epochs = args.number_epochs,
                    shuffle = False,
                    verbose= 1 if rank == 0 else 0,
                    validation_data=val_dataset,
                    callbacks=[] if rank == 0 else [],)

# EVALUATION
#if rank == 0:
tnt_loss_accuracy = model.evaluate(test_dataset,
                                   verbose=0)
print("Tarantella[test_loss, accuracy] = ", tnt_loss_accuracy)
print("Reference [test_loss, accuracy] = ", reference_loss_accuracy)

