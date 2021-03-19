import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress INFO messages from tf

from utilities import *
import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses
import tarantella.keras.metrics as tnt_metrics

comm_size = tnt.get_size()
assert(comm_size == 2)

rank = tnt.get_rank()
p_0_rank = 0
p_1_rank = 1
master_rank = p_1_rank # to test final accuracies on same rank

number_tags = 2 # [micro_batch_id, connection_id]

### argument parsing
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
reference_x = layers.Dense(fc_units, activation='relu', name='dense_relu')(reference_x)
reference_output = layers.Dense(num_mnist_classes,
                                activation='softmax',
                                name='dense_softmax')(reference_x + reference_x)
reference_model = keras.Model(inputs=reference_input, outputs=reference_output, name="reference_model")

# datasets
train_dataset_reference = create_dataset_from_arrays(x_train, y_train, batch_size=batch_size) \
                          .shuffle(len(x_train), shuffle_seed)
val_dataset_reference = create_dataset_from_arrays(x_val, y_val, batch_size=batch_size)
test_dataset_reference = create_dataset_from_arrays(x_test, y_test, batch_size=batch_size)

## pipelined model
num_partitions = 2
assert comm_size == num_partitions

p_0_id = 0
p_1_id = 1
size_of_float = 4
partition_table = { 0 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * size_of_float),
                    1 : ((p_0_rank, p_1_rank), fc_units * micro_batch_size * size_of_float) }

ppl_comm = tnt.PipelineCommunicator(partition_table, num_micro_batches)

# --- core model on partition 0
tf.random.set_seed(42) # reset seed, so initial weights are same as for the reference model
p_0_core_input = keras.Input(shape=(28,28,1,)) # may be more than one
p_0_core_x = layers.Flatten()(p_0_core_input)
p_0_core_output_0 = layers.Dense(fc_units, activation='relu', name='dense_relu'+'_0')(p_0_core_x)
p_0_core_output_1 = tnt_layers.IdentityLayer(name='dense_relu'+'_1')(p_0_core_output_0)
p_0_core = keras.Model(inputs=p_0_core_input,
                       outputs=[p_0_core_output_0, p_0_core_output_1],
                       name="core_layers_p_0")

# --- core model on partition 1
p_1_core_input_0_shape = p_0_core.outputs[0].shape[1:]
p_1_core_input_1_shape = p_0_core.outputs[1].shape[1:]
p_1_core_input_0 = keras.Input(shape=p_1_core_input_0_shape) # TODO: Maybe add dtypes?
p_1_core_input_1 = keras.Input(shape=p_1_core_input_1_shape)
p_1_core_x = p_1_core_input_0 + p_1_core_input_1
p_1_core_output_0 = layers.Dense(num_mnist_classes, activation='softmax', name='dense_softmax')(p_1_core_x)
p_1_core = keras.Model(inputs=[p_1_core_input_0, p_1_core_input_1],
                       outputs=[p_1_core_output_0],
                       name="core_layers_p_1")

# --- shared model on partition 0
p_0_shared_input_0 = keras.Input(shape=(28,28,1,))
p_0_shared_send_tag_0 = keras.Input(shape=(number_tags,), dtype=tf.int32)
p_0_shared_send_tag_1 = keras.Input(shape=(number_tags,), dtype=tf.int32)
p_0_shared_input_seq = keras.Input(shape=(1,)) # use to model sequential dependencies # TODO: Add dtype?!
  
p_0_shared_core_inputs = [p_0_shared_input_0]
p_0_shared_recv_tags = [] 
p_0_shared_send_tags = [p_0_shared_send_tag_0, p_0_shared_send_tag_1]
p_0_shared_start_seq = [p_0_shared_input_seq]

p_0_shared_x = tnt_layers.RemoveSeqInput()(p_0_shared_core_inputs + p_0_shared_start_seq)
p_0_shared_x = p_0_core(p_0_shared_x)
p_0_shared_output_0 = tnt_layers.SendLayer(pipeline_communicator = ppl_comm)(p_0_shared_x[0], p_0_shared_send_tags[0])
p_0_shared_output_1 = tnt_layers.SendLayer(pipeline_communicator = ppl_comm)(p_0_shared_x[1], p_0_shared_send_tags[1])
p_0_shared_outputs = [p_0_shared_output_0, p_0_shared_output_1]
p_0_shared_outputs = tnt_layers.AddSeqOutput(micro_batch_size = micro_batch_size)(p_0_shared_outputs)
p_0_shared_inputs = p_0_shared_core_inputs + p_0_shared_recv_tags + p_0_shared_send_tags + p_0_shared_start_seq
p_0_shared_model = keras.Model(inputs=p_0_shared_inputs, outputs=p_0_shared_outputs, name="p_0_shared")

# --- shared model on partition 1
p_1_shared_input_0 = keras.Input(shape = p_1_core.inputs[0].shape[1:])
p_1_shared_input_1 = keras.Input(shape = p_1_core.inputs[1].shape[1:])
p_1_shared_recv_tag_0 = keras.Input(shape=(number_tags,), dtype=tf.int32)
p_1_shared_recv_tag_1 = keras.Input(shape=(number_tags,), dtype=tf.int32)
p_1_shared_input_seq = keras.Input(shape=(1,))

p_1_shared_core_inputs = [p_1_shared_input_0, p_1_shared_input_1]
p_1_shared_recv_tags = [p_1_shared_recv_tag_0, p_1_shared_recv_tag_1] 
p_1_shared_send_tags = []
p_1_shared_start_seq = [p_1_shared_input_seq]

p_1_shared_x = tnt_layers.RemoveSeqInput()(p_1_shared_core_inputs + p_1_shared_start_seq)
p_1_shared_recved_0 = tnt_layers.RecvLayer(pipeline_communicator = ppl_comm)(p_1_shared_x[0], p_1_shared_recv_tags[0])
p_1_shared_recved_1 = tnt_layers.RecvLayer(pipeline_communicator = ppl_comm)(p_1_shared_x[1], p_1_shared_recv_tags[1])

p_1_shared_outputs = p_1_core([p_1_shared_recved_0, p_1_shared_recved_1])
p_1_shared_outputs = tnt_layers.AddSeqOutput(micro_batch_size=micro_batch_size)(p_1_shared_outputs)
p_1_shared_inputs = p_1_shared_core_inputs + p_1_shared_recv_tags + p_1_shared_send_tags + p_1_shared_start_seq
p_1_shared_model = keras.Model(inputs=p_1_shared_inputs, outputs=p_1_shared_outputs, name="p_1_shared")

# --- microbatched model on partition 0
p_0_shared_m_0_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_0_i_0")
p_0_shared_m_1_input_0 = keras.Input(shape=(28,28,1,), name="p_0_m_1_i_0")
p_0_shared_m_0_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_0_s_0", dtype=tf.int32)
p_0_shared_m_0_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_0_s_1", dtype=tf.int32)
p_0_shared_m_1_send_tag_0 = keras.Input(shape=(number_tags,), name="p_0_m_1_s_0", dtype=tf.int32)
p_0_shared_m_1_send_tag_1 = keras.Input(shape=(number_tags,), name="p_0_m_1_s_1", dtype=tf.int32)
p_0_shared_input_seq = keras.Input(shape=(1,), name = "p_0_start_seq")
  
p_0_shared_core_inputs = [p_0_shared_m_0_input_0, p_0_shared_m_1_input_0]
p_0_shared_recv_tags = [] 
p_0_shared_send_tags = [p_0_shared_m_0_send_tag_0, p_0_shared_m_0_send_tag_1,
                        p_0_shared_m_1_send_tag_0, p_0_shared_m_1_send_tag_1]
p_0_shared_start_seq = [p_0_shared_input_seq]

p_0_m_0_outputs = p_0_shared_model(p_0_shared_core_inputs[0:1] + p_0_shared_send_tags[0:2] + p_0_shared_start_seq)
p_0_m_1_outputs = p_0_shared_model(p_0_shared_core_inputs[1:2] + p_0_shared_send_tags[2:4] + [p_0_m_0_outputs[-1]])

p_0_outputs = p_0_m_0_outputs[:-1] + p_0_m_1_outputs[:-1] + [p_0_m_1_outputs[-1]]
p_0_outputs[0] = tnt_layers.IdentityLayer(name="p_0_m_0_o_0")(p_0_outputs[0])
p_0_outputs[1] = tnt_layers.IdentityLayer(name="p_0_m_0_o_1")(p_0_outputs[1])
p_0_outputs[2] = tnt_layers.IdentityLayer(name="p_0_m_1_o_0")(p_0_outputs[2])
p_0_outputs[3] = tnt_layers.IdentityLayer(name="p_0_m_1_o_1")(p_0_outputs[3])
p_0_outputs[4] = tnt_layers.IdentityLayer(name="p_0_end_seq")(p_0_outputs[4])
p_0_inputs = p_0_shared_core_inputs + p_0_shared_recv_tags + p_0_shared_send_tags + p_0_shared_start_seq
p_0 = keras.Model(inputs=p_0_inputs, outputs=p_0_outputs, name="p_0")

# --- microbatched model on partition 1
p_1_shared_m_0_input_0 = keras.Input(shape=p_1_core.inputs[0].shape[1:], name="p_1_m_0_i_0")
p_1_shared_m_0_input_1 = keras.Input(shape=p_1_core.inputs[1].shape[1:], name="p_1_m_0_i_1")
p_1_shared_m_1_input_0 = keras.Input(shape=p_1_core.inputs[0].shape[1:], name="p_1_m_1_i_0")
p_1_shared_m_1_input_1 = keras.Input(shape=p_1_core.inputs[1].shape[1:], name="p_1_m_1_i_1")
p_1_shared_m_0_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_0_r_0", dtype=tf.int32)
p_1_shared_m_0_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_0_r_1", dtype=tf.int32)
p_1_shared_m_1_recv_tag_0 = keras.Input(shape=(number_tags,), name="p_1_m_1_r_0", dtype=tf.int32)
p_1_shared_m_1_recv_tag_1 = keras.Input(shape=(number_tags,), name="p_1_m_1_r_1", dtype=tf.int32)
p_1_shared_input_seq = keras.Input(shape=(1,), name = "p_1_start_seq")

p_1_shared_core_inputs = [p_1_shared_m_0_input_0, p_1_shared_m_0_input_1,
                          p_1_shared_m_1_input_0, p_1_shared_m_1_input_1]
p_1_shared_recv_tags = [p_1_shared_m_0_recv_tag_0, p_1_shared_m_0_recv_tag_1,
                        p_1_shared_m_1_recv_tag_0, p_1_shared_m_1_recv_tag_1]
p_1_shared_send_tags = []
p_1_shared_start_seq = [p_1_shared_input_seq]

p_1_m_0_outputs = p_1_shared_model(p_1_shared_core_inputs[0:2] + p_1_shared_recv_tags[0:2] + p_1_shared_start_seq)
p_1_m_1_outputs = p_1_shared_model(p_1_shared_core_inputs[2:4] + p_1_shared_recv_tags[2:4] + [p_1_m_0_outputs[-1]])

p_1_outputs = p_1_m_0_outputs[:-1] + p_1_m_1_outputs[:-1] + [p_1_m_1_outputs[-1]]
p_1_outputs[0] = tnt_layers.IdentityLayer(name="p_1_m_0_o_0")(p_1_outputs[0])
p_1_outputs[1] = tnt_layers.IdentityLayer(name="p_1_m_1_o_0")(p_1_outputs[1])
p_1_outputs[2] = tnt_layers.IdentityLayer(name="p_1_end_seq")(p_1_outputs[2])
p_1_inputs = p_1_shared_core_inputs + p_1_shared_recv_tags + p_1_shared_send_tags + p_1_shared_start_seq
p_1 = keras.Model(inputs=p_1_inputs, outputs=p_1_outputs, name="p_1")

# datasets

# TODO:
# Does the fake datasets/labels need a finite size, when used in a "middle" partition,
# so `fit` would know, when to stop training?
x_comm_0 = tf.data.Dataset.from_tensors(np.zeros(p_1_core.inputs[0].shape[1:])).repeat()
x_comm_1 = tf.data.Dataset.from_tensors(np.zeros(p_1_core.inputs[1].shape[1:])).repeat()
y_zero_loss = tf.data.Dataset.from_tensors(np.zeros(1)).repeat()

p_0_recv_connection_ids = [] # recv connection ids for each RecvLayer
p_0_send_connection_ids = [0, 1] # send connection ids for each SendLayer
p_0_train_dataset = create_micro_batched_dataset(samples = [tf.data.Dataset.from_tensor_slices(x_train)],
                                                 labels = [y_zero_loss, y_zero_loss],
                                                 recv_connection_ids = p_0_recv_connection_ids, 
                                                 send_connection_ids = p_0_send_connection_ids, 
                                                 partition_id = p_0_id,
                                                 num_micro_batches = num_micro_batches,
                                                 micro_batch_size = micro_batch_size) \
                    .shuffle(len(x_train), shuffle_seed)
p_0_val_dataset = create_micro_batched_dataset(samples = [tf.data.Dataset.from_tensor_slices(x_val)],
                                               labels = [y_zero_loss, y_zero_loss],
                                               recv_connection_ids = p_0_recv_connection_ids, 
                                               send_connection_ids = p_0_send_connection_ids, 
                                               partition_id = p_0_id,
                                               num_micro_batches = num_micro_batches,
                                               micro_batch_size = micro_batch_size)
p_0_test_dataset = create_micro_batched_dataset(samples = [tf.data.Dataset.from_tensor_slices(x_test)],
                                                labels = [y_zero_loss, y_zero_loss],
                                                recv_connection_ids = p_0_recv_connection_ids, 
                                                send_connection_ids = p_0_send_connection_ids, 
                                                partition_id = p_0_id,
                                                num_micro_batches = num_micro_batches,
                                                micro_batch_size = micro_batch_size)

p_1_recv_connection_ids = [0, 1]
p_1_send_connection_ids = []
p_1_train_dataset = create_micro_batched_dataset(samples = [x_comm_0, x_comm_1],
                                                 labels = [tf.data.Dataset.from_tensor_slices(y_train)],
                                                 recv_connection_ids = p_1_recv_connection_ids,
                                                 send_connection_ids = p_1_send_connection_ids,
                                                 partition_id = p_1_id,
                                                 num_micro_batches = num_micro_batches,
                                                 micro_batch_size = micro_batch_size) \
                    .shuffle(len(x_train), shuffle_seed)
p_1_val_dataset = create_micro_batched_dataset(samples = [x_comm_0, x_comm_1],
                                               labels = [tf.data.Dataset.from_tensor_slices(y_val)],
                                               recv_connection_ids = p_1_recv_connection_ids,
                                               send_connection_ids = p_1_send_connection_ids,
                                               partition_id = p_1_id,
                                               num_micro_batches = num_micro_batches,
                                               micro_batch_size = micro_batch_size)
p_1_test_dataset = create_micro_batched_dataset(samples = [x_comm_0, x_comm_1],
                                                labels = [tf.data.Dataset.from_tensor_slices(y_test)],
                                                recv_connection_ids = p_1_recv_connection_ids,
                                                send_connection_ids = p_1_send_connection_ids,
                                                partition_id = p_1_id,
                                                num_micro_batches = num_micro_batches,
                                                micro_batch_size = micro_batch_size)

# write model descriptions to files for debugging
callbacks = []
if user_verbosity != 0:
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/rank" + str(rank)
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  callbacks += [tensorboard_callback]

if rank == 0:
  keras.utils.plot_model(reference_model, "reference_model.png", show_shapes=True)
  keras.utils.plot_model(p_0_core, "p_0_core.png", show_shapes=True)
  keras.utils.plot_model(p_0_shared_model, "p_0_shared_model.png", show_shapes=True)
  keras.utils.plot_model(p_0, "p_0.png", show_shapes=True)
  keras.utils.plot_model(p_1_core, "p_1_core.png", show_shapes=True)
  keras.utils.plot_model(p_1_shared_model, "p_1_shared_model.png", show_shapes=True)
  keras.utils.plot_model(p_1, "p_1.png", show_shapes=True)

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

# pipelined model
if rank == p_0_rank:
  p_0_losses = {"p_0_m_0_o_0" : tnt_losses.ZeroLoss(),
                "p_0_m_0_o_1" : tnt_losses.ZeroLoss(),
                "p_0_m_1_o_0" : tnt_losses.ZeroLoss(),
                "p_0_m_1_o_1" : tnt_losses.ZeroLoss(),
                "p_0_end_seq" : tnt_losses.ZeroLoss()}
  p_0.compile(optimizer = sgd,
              loss = p_0_losses)
  p_0.fit(p_0_train_dataset,
          validation_data = p_0_val_dataset,
          epochs = number_epochs,
          verbose = 0,
          callbacks = callbacks)
  # p_0.evaluate(p_0_test_dataset,
  #              verbose = 0)
if rank == p_1_rank:
  print("\nTraining pipelined model")
  p_1_losses = {"p_1_m_0_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                "p_1_m_1_o_0" : keras.losses.SparseCategoricalCrossentropy(),
                "p_1_end_seq" : tnt_losses.ZeroLoss()}
  p_1_loss_weights = {"p_1_m_0_o_0" : 1./num_micro_batches,
                      "p_1_m_1_o_0" : 1./num_micro_batches,
                      "p_1_end_seq" : 0.}
  p_1_metrics = {"p_1_m_0_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                 "p_1_m_1_o_0" : keras.metrics.SparseCategoricalAccuracy(),
                 "p_1_end_seq" : tnt_metrics.ZeroMetric()}
  p_1.compile(optimizer = sgd,
              loss = p_1_losses,
              loss_weights = p_1_loss_weights,
              metrics = p_1_metrics)
  p_1.fit(p_1_train_dataset,
          validation_data = p_1_val_dataset,
          epochs = number_epochs,
          verbose = user_verbosity,
          callbacks = callbacks)
  # pipelining_results = p_1.evaluate(p_1_test_dataset,
  #                                   verbose = user_verbosity)
  print("Reference model results: [%.8f, %.8f]" % (single_rank_results[0], single_rank_results[1]))
  # print("Pipelined model results: [%.8f, %.8f]" % (pipelining_results[0],
  #                                                  (pipelining_results[-2]+pipelining_results[-3])/2))