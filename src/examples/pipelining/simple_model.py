from utilities import *
from communication_layers import *
import dataset_validation as ds
import pipelining_model as tntmodel

import os
import datetime
import numpy as np
from mpi4py import MPI

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress INFO messages from tf

tf.config.threading.set_inter_op_parallelism_threads(4)

### MPI setup & argument parsing
context = MPI.COMM_WORLD
comm_size = context.Get_size()
assert(comm_size == 2)
rank = context.Get_rank()
partition_0_rank = 0
partition_1_rank = 1
master_rank = partition_0_rank # to test final accuracies on same rank

args = parse_args()
num_micro_batches = args.num_micro_batches
batch_size = args.batch_size 
micro_batch_size = batch_size // num_micro_batches
number_epochs = args.number_epochs
train_size = args.train_size
val_size = args.val_size
test_size = args.test_size
user_verbosity = args.verbose
assert(batch_size == micro_batch_size * num_micro_batches)

### DATA LOADING
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_as_np_arrays(train_size, val_size, test_size)

### MODEL CONSTRUCTION (on _all_ ranks)
fc_units = 100
num_mnist_classes = 10
shuffle_seed = 17
tf.random.set_seed(42)

## reference model
# topology
reference_input = keras.Input(shape=(28,28,1,), name='reference_input')
reference_x = layers.Flatten()(reference_input)
reference_x = layers.Dense(fc_units, activation='relu', name='dense_relu1')(reference_x)
reference_y = layers.Dense(fc_units, activation='relu', name='dense_relu2')(reference_x)
reference_output = layers.Dense(num_mnist_classes,
                                activation='softmax',
                                name='dense_softmax')(reference_x + reference_y)
reference_model = keras.Model(inputs=reference_input, outputs=reference_output, name="reference_model")

# datasets
train_dataset_reference = create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                          .shuffle(len(x_train), shuffle_seed)
val_dataset_reference = create_dataset_from_arrays(x_val, y_val, batch_size=batch_size)
test_dataset_reference = create_dataset_from_arrays(x_test, y_test, batch_size=batch_size)

## pipelined model
# partition ids
num_partitions = 2
partition_0_id = 0
partition_1_id = 1
assert comm_size == num_partitions
tf.random.set_seed(42) # reset seed, so initial weights are same as for the reference model

# --- core model on partition 0
partition_0_core_input = keras.Input(shape=(28,28,1,)) # may be more than one
partition_0_core_x = layers.Flatten()(partition_0_core_input)
partition_0_core_output_0 = layers.Dense(fc_units, activation='relu', name='dense_relu1'+'_0')(partition_0_core_x)
partition_0_core_output_1 = IdentityLayer(name='dense_relu1'+'_1')(partition_0_core_output_0)
partition_0_core = keras.Model(inputs=partition_0_core_input,
                               outputs=[partition_0_core_output_0, partition_0_core_output_1],
                               name="core_layers_partition_0")

# --- core model on partition 1
partition_1_input_0_shape = partition_0_core.outputs[0].shape
partition_1_input_1_shape = partition_0_core.outputs[1].shape
partition_1_input_0 = keras.Input(shape=partition_1_input_0_shape[1:]) # pass w/o batch size!
partition_1_input_1 = keras.Input(shape=partition_1_input_1_shape[1:])

partition_1_core_inputs = [partition_1_input_0, partition_1_input_1]
partition_1_core_x = layers.Dense(fc_units, activation='relu', name='dense_relu2')(partition_1_core_inputs[0])
partition_1_core_outputs = layers.Dense(num_mnist_classes,
                                        activation='softmax',
                                        name='dense_softmax')(partition_1_core_inputs[1] + partition_1_core_x)
partition_1_core = keras.Model(inputs=partition_1_core_inputs,
                               outputs=partition_1_core_outputs,
                               name="core_layers_partition_1")

# partition interconnection info
# partition table: { partition_id : (num inputs, num outputs) }
partition_table = {0 : tntmodel.PartitionInfo(num_inputs = 1, num_outputs = 2),
                   1 : tntmodel.PartitionInfo(num_inputs = 2, num_outputs = 1)}

partition_0_output_0_to_partition_1_input_0 = tntmodel.CommunicationInfo(src_partition = 0, dest_partition = 1,
                                                                         src_output_index = 0, dest_input_index = 0)
partition_0_output_1_to_partition_1_input_1 = tntmodel.CommunicationInfo(src_partition = 0, dest_partition = 1,
                                                                         src_output_index = 1, dest_input_index = 1)

# partition_*_input_info: { input_id : CommunicationInfo/NoCommunication }
# partition_*_output_info: { output_id : CommunicationInfo/NoCommunication }
partition_0_input_info = {0 : tntmodel.NoCommunication()}
partition_0_output_info = {0 : partition_0_output_0_to_partition_1_input_0,
                           1 : partition_0_output_1_to_partition_1_input_1}
partition_0_losses = { } # could be non-empty if calculating some real loss

partition_1_input_info = {0 : partition_0_output_0_to_partition_1_input_0,
                          1 : partition_0_output_1_to_partition_1_input_1}
partition_1_output_info = {0 : tntmodel.NoCommunication()}
partition_1_losses = {0 : keras.losses.SparseCategoricalCrossentropy()}
partition_1_metrics = {0: keras.metrics.SparseCategoricalAccuracy()}

#--------------------------

# --- model on partition 0
model_generator0 = tntmodel.MicrobatchedModelGenerator(partition_0_core, 
                                                     micro_batch_size, num_micro_batches,
                                                     partition_id = partition_0_id,
                                                     context = context,
                                                     partition_table = partition_table,
                                                     input_info = partition_0_input_info,
                                                     output_info = partition_0_output_info)
partition_0 = model_generator0.get_model()

# --- model on partition 1
model_generator1 = tntmodel.MicrobatchedModelGenerator(partition_1_core,
                                                     micro_batch_size, num_micro_batches,
                                                     partition_id = partition_1_id,
                                                     context = context,
                                                     partition_table = partition_table,
                                                     input_info = partition_1_input_info,
                                                     output_info = partition_1_output_info)
partition_1 = model_generator1.get_model()

# TODO: move datasets creation to model generator (create fake datasets there)
## datasets
# empty buffers for communication
x_train_fake = tf.data.Dataset.from_tensor_slices(np.zeros(train_size))
y_train_fake = tf.data.Dataset.from_tensor_slices(np.zeros(train_size))
x_val_fake = tf.data.Dataset.from_tensor_slices(np.zeros(val_size))
y_val_fake = tf.data.Dataset.from_tensor_slices(np.zeros(val_size))
x_test_fake = tf.data.Dataset.from_tensor_slices(np.zeros(test_size))
y_test_fake = tf.data.Dataset.from_tensor_slices(np.zeros(test_size))
# datasets on partition 0
partition_0_train_dataset = create_micro_batched_dataset([tf.data.Dataset.from_tensor_slices(x_train)],
                                                         [y_train_fake, y_train_fake],
                                                         partition_0_id,
                                                         num_micro_batches,
                                                         micro_batch_size) \
                            .shuffle(len(x_train), shuffle_seed)
partition_0_val_dataset = create_micro_batched_dataset([tf.data.Dataset.from_tensor_slices(x_val)],
                                                       [y_val_fake, y_val_fake],
                                                       partition_0_id,
                                                       num_micro_batches,
                                                       micro_batch_size)
partition_0_test_dataset = create_micro_batched_dataset([tf.data.Dataset.from_tensor_slices(x_test)],
                                                        [y_test_fake, y_test_fake],
                                                        partition_0_id,
                                                        num_micro_batches,
                                                        micro_batch_size)
# datasets on partition 1
partition_1_train_dataset = create_micro_batched_dataset([x_train_fake, x_train_fake],
                                                         [tf.data.Dataset.from_tensor_slices(y_train)],
                                                         partition_1_id,
                                                         num_micro_batches,
                                                         micro_batch_size) \
                            .shuffle(len(x_train), shuffle_seed)
partition_1_val_dataset = create_micro_batched_dataset([x_val_fake, x_val_fake],
                                                       [tf.data.Dataset.from_tensor_slices(y_val)],
                                                       partition_1_id,
                                                       num_micro_batches,
                                                       micro_batch_size)
partition_1_test_dataset = create_micro_batched_dataset([x_test_fake, x_test_fake],
                                                        [tf.data.Dataset.from_tensor_slices(y_test)],
                                                        partition_1_id,
                                                        num_micro_batches,
                                                        micro_batch_size)

callbacks = []
# write model descriptions to files for debugging
if rank == master_rank and user_verbosity != 0:
  reference_model.summary()
  partition_0_core.summary()
  partition_1_core.summary()
  partition_0.summary()
  partition_1.summary()

  keras.utils.plot_model(reference_model, "reference_model.png", show_shapes=True)
  keras.utils.plot_model(partition_0_core, "partition_0_core.png", show_shapes=True)
  keras.utils.plot_model(partition_1_core, "partition_1_core.png", show_shapes=True)
  
  keras.utils.plot_model(partition_0, "partition_0_generated.png", show_shapes=True)
  keras.utils.plot_model(partition_1, "partition_1_generated.png", show_shapes=True)

  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  callbacks += [tensorboard_callback]

### MODEL COMPILE/TRAIN/TEST (on each rank individually)
sgd = keras.optimizers.SGD(learning_rate=args.learning_rate)

# single rank model
if rank == master_rank:
  print("\nTraining reference model")
  reference_model.compile(optimizer = sgd,
                          loss = keras.losses.SparseCategoricalCrossentropy(),
                          metrics = [keras.metrics.SparseCategoricalAccuracy()],
                          experimental_run_tf_function = False)
  reference_model.fit(train_dataset_reference,
                      validation_data = val_dataset_reference,
                      epochs = number_epochs,
                      verbose = user_verbosity)
  single_rank_results = reference_model.evaluate(test_dataset_reference,
                                                 verbose = user_verbosity)
  print("Reference model results: [%.8f, %.8f]" % (single_rank_results[0], single_rank_results[1]))

context.Barrier()

losses = [
          CommLoss(context, fwd_tag = 0, bwd_tag = 1, src_dest= 1, name='comm_loss0'),
          CommLoss(context, fwd_tag = 2, bwd_tag = 3, src_dest= 1, name='comm_loss1'),
          CommLoss(context, fwd_tag = 4, bwd_tag = 5, src_dest= 1, name='comm_loss2'),
          CommLoss(context, fwd_tag = 6, bwd_tag = 7, src_dest= 1, name='comm_loss3'),
          ZeroLoss()]

p1_losses = [
          RealLoss(context,  name='real_loss0'),
          RealLoss(context,  name='real_loss1'),
          ZeroLoss()]

# pipelined model
if rank == partition_0_rank:
  partition_0.compile(optimizer = sgd,
                      loss = losses,
#                      loss = model_generator0.get_losses(partition_0_losses),
#                      loss_weights = model_generator0.get_loss_weights({}),
#                      metrics = model_generator0.get_metrics({}),
                      experimental_run_tf_function = False)
  partition_0.fit(partition_0_train_dataset,
                  validation_data = partition_0_val_dataset,
                  epochs = number_epochs,
                  verbose = 0)
  partition_0.evaluate(partition_0_test_dataset,
                       verbose = 0)
if rank == partition_1_rank:
  print("\nTraining pipelined model")
  partition_1.compile(optimizer = sgd,
                      loss = p1_losses,
                      #loss = model_generator1.get_losses(partition_1_losses),
                      #loss_weights = model_generator1.get_loss_weights({}),
                      metrics = model_generator1.get_metrics(partition_1_metrics),
                      experimental_run_tf_function = False)
  partition_1.fit(partition_1_train_dataset,
                  validation_data = partition_1_val_dataset,
                  epochs = number_epochs,
                  verbose = user_verbosity,
                  callbacks = callbacks)
  pipelining_results = partition_1.evaluate(partition_1_test_dataset,
                                            verbose = user_verbosity)
  print("Pipelined model results: ", pipelining_results)
