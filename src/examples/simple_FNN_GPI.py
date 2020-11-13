import argparse
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.utils import keras_utils as utils

import tarantella as tnt


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
  return ds.batch(batch_size, drop_remainder=True)

args = parse_args()

tnt.init(args.ngpus_per_node)
rank = tnt.get_rank()
comm_size = tnt.get_size()

batch_size = args.batch_size
micro_batch_size = args.batch_size // comm_size
train_size = args.train_size
val_size = args.val_size
test_size = args.test_size
shuffle_seed = 12

# MODELS
# ======

# Reference model
tf.random.set_seed(42)
inputs = keras.Input(shape=(28,28,1,), name='input')
x = layers.Flatten()(inputs)
x = layers.Dense(200, activation='relu', name='FC')(x)
x = layers.Dense(200, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax', name='softmax')(x)
reference_model = keras.Model(inputs=inputs, outputs=outputs)

# Tarantella model
tf.random.set_seed(42) # re-seed to get the same initial weights as in reference
tnt_model = tnt.models.clone_model(reference_model)

# Optimizer
sgd_reference = keras.optimizers.SGD(learning_rate = args.learning_rate)
sgd_tnt = keras.optimizers.SGD(learning_rate = args.learning_rate)

# COMPILE
# =======

reference_model.compile(optimizer = sgd_reference,
                        loss=keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[keras.metrics.SparseCategoricalAccuracy()],
                       )

tnt_model.compile(optimizer = sgd_tnt,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                 )

# DATASET
# =======

(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_as_np_arrays(train_size, val_size, test_size)
train_dataset = create_dataset_from_arrays(x_train, y_train, batch_size).shuffle(len(x_train), shuffle_seed)
val_dataset = create_dataset_from_arrays(x_val, y_val, batch_size)
test_dataset = create_dataset_from_arrays(x_test, y_test, batch_size)

# TRAINING
# ========

# Callbacks
# ---------

# TensorBoard
log_dir = "/home/labus/git/hpdlf/src/examples/logs"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ModelCheckpoint
save_weights_only = False
if save_weights_only:
  chk_path = "/home/labus/git/hpdlf/src/examples/checkpoints/weights"
else:
  chk_path = "/home/labus/git/hpdlf/src/examples/checkpoints/chk_{epoch}"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    chk_path, monitor='val_acc', verbose=1, save_best_only=False,
    save_weights_only=save_weights_only, mode='auto', save_freq='epoch', options=None)

# LearningRateScheduler
def scheduler(epoch, learning_rate):
  return learning_rate * tf.math.exp(-0.1)
learning_rate_scheduler_callback_reference = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 0)
learning_rate_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)

# Reference model
# ---------------
history = reference_model.fit(train_dataset,
                              epochs = args.number_epochs,
                              shuffle = False,
                              verbose = args.verbose if tnt.is_master_rank() else 0,
                              validation_data = val_dataset,
                              callbacks = [learning_rate_scheduler_callback_reference],
                             )
reference_loss_accuracy = reference_model.evaluate(test_dataset,
                                                   verbose=0)

# Tarantella model
# ----------------
history = tnt_model.fit(train_dataset,
                        epochs = args.number_epochs,
                        shuffle = False,
                        verbose = args.verbose,
                        validation_data = val_dataset,
                        callbacks = [model_checkpoint_callback,
                                     learning_rate_scheduler_callback,
                                     tensorboard_callback],
                       )
tnt_loss_accuracy = tnt_model.evaluate(test_dataset, verbose=0)

if tnt.is_master_rank():
  print("Reference [test_loss, accuracy] = ", reference_loss_accuracy)
  print("Tarantella[test_loss, accuracy] = ", tnt_loss_accuracy)

# RECONSTRUCT SAVED MODEL
# =======================

sgd_reconstructed = keras.optimizers.SGD(learning_rate = args.learning_rate)

# Save tnt.Model configuration, initialize new model, and load weights from checkpoint
if save_weights_only:
  config = tnt_model.get_config()
  reconstructed_model = tnt.models.model_from_config(config)
  reconstructed_model.compile(optimizer = sgd_reconstructed,
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy()],
                   )
  reconstructed_model.load_weights(filepath = chk_path)

# Restore Tarantella model from checkpoint
if not save_weights_only:
  model_path = "/home/labus/git/hpdlf/src/examples/checkpoints/chk_" + str(args.number_epochs)
  reconstructed_model = tnt.models.load_model(model_path)
  # TODO: Automatically compile in `tnt.models.load_model`
  reconstructed_model.compile(optimizer = sgd_reconstructed,
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()],
                             )

reconstructed_loss_accuracy = reconstructed_model.evaluate(test_dataset, verbose=0)

if tnt.is_master_rank():
  print("Reconstructed model [test_loss, accuracy] = ", reconstructed_loss_accuracy)
