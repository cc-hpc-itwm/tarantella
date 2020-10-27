import argparse
import re

import tensorflow as tf
from tensorflow import keras
from enum import Enum

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-bs", "--batch_size", type=int, default=64)
  parser.add_argument("-micro", "--num_micro_batches", type=int, default=2)
  parser.add_argument("-e", "--number_epochs", type=int, default=1)
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
  parser.add_argument("-train", "--train_size", type=int, default=50000)
  parser.add_argument("-val", "--val_size", type=int, default=10000)
  parser.add_argument("-test", "--test_size", type=int, default=10000)
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

class InoutLayerType(Enum):
    input = 'i'
    output = 'o'
    start_seq = 'start_seq'
    end_seq = 'end_seq'
    recv_tag = 'r'
    send_tag = 's'

def create_name_partition(partition_id):
  return 'p_%s' % (str(partition_id))

def create_name_micro_batched_layer(partition_id, element_type, layer_id = None, micro_batch_id = None):
  """Create a name for a layer of `type` either input/output, based on the partition_id, 
  layer_id, and an optional micro_batch_id
  """
  if not isinstance(element_type, InoutLayerType):
    raise TypeError("[create_name_micro_batched_layer] Layer type should be an `InoutLayerType` object")
  if not element_type in [InoutLayerType.start_seq, InoutLayerType.end_seq]:
    name = 'p_%s' % (str(partition_id))
    if micro_batch_id is not None:
      name = name + '_m_%s' % (str(micro_batch_id))
    name = name + '_%s_%s' % (element_type.value, str(layer_id))

  else:
    if micro_batch_id is not None:
      raise ValueError("[create_name_micro_batched_layer] micro_batch_id should be `None` for layer type `*_seq`")
    if layer_id is not None:
      raise ValueError("[create_name_micro_batched_layer] layer_id should be `None` for layer type `*_seq`")
    name = 'p_%s_%s' % (str(partition_id), element_type.value)
  return name

def create_names_micro_batched_model(partition_id, num_core_elements, element_type, num_micro_batches):
  names = list()
  for i in range(num_core_elements):
    for m in range(num_micro_batches):
      names.append(create_name_micro_batched_layer(partition_id, element_type, i, m))
  return names

# TODO: move to model generator and make sure the order matches with input construction
def create_micro_batched_dataset(samples, labels, recv_connection_ids, send_connection_ids,
                                 partition_id, num_micro_batches, micro_batch_size):
  assert isinstance(samples, list) and len(samples) > 0
  assert isinstance(labels, list) and len(labels) > 0
  num_inputs = len(samples)
  num_outputs = len(labels)
  num_recvs = len(recv_connection_ids)
  num_sends = len(send_connection_ids)

  input_names = create_names_micro_batched_model(partition_id=partition_id, 
                                                 num_core_elements=num_inputs,
                                                 element_type=InoutLayerType.input, 
                                                 num_micro_batches=num_micro_batches)
  recv_tag_names = create_names_micro_batched_model(partition_id=partition_id, 
                                                    num_core_elements=num_recvs,
                                                    element_type=InoutLayerType.recv_tag, 
                                                    num_micro_batches=num_micro_batches)
  send_tag_names = create_names_micro_batched_model(partition_id=partition_id, 
                                                    num_core_elements=num_sends,
                                                    element_type=InoutLayerType.send_tag, 
                                                    num_micro_batches=num_micro_batches)
  start_seq_name = create_name_micro_batched_layer(partition_id=partition_id,
                                                   element_type=InoutLayerType.start_seq)
  output_names = create_names_micro_batched_model(partition_id=partition_id, 
                                                  num_core_elements=num_outputs,
                                                  element_type=InoutLayerType.output, 
                                                  num_micro_batches=num_micro_batches)
  end_seq_name = create_name_micro_batched_layer(partition_id=partition_id,
                                                 element_type=InoutLayerType.end_seq)
  
  # create micro-batched inputs/outputs from original samples/labels and
  # an additional placeholder dataset each, for sequential micro-batch execution
  input_datasets = list()
  for i in range(num_inputs):
    for m in range(num_micro_batches):
      input_datasets.append(samples[i].batch(micro_batch_size).shard(num_micro_batches, m))
  
  recv_tag_datasets = list()
  for connection_id in recv_connection_ids:
    for m in range(num_micro_batches):
      recv_tag_datasets.append(tf.data.Dataset.from_tensors(tf.constant([m, connection_id], dtype=tf.int32)).batch(1).repeat())

  send_tag_datasets = list()
  for connection_id in send_connection_ids:
    for m in range(num_micro_batches):
      send_tag_datasets.append(tf.data.Dataset.from_tensors(tf.constant([m, connection_id], dtype=tf.int32)).batch(1).repeat())

  start_seq_dataset = [tf.data.Dataset.from_tensors(tf.constant(0.0)).batch(1).repeat()]

  output_datasets = list()
  for o in range(num_outputs):
    for m in range(num_micro_batches):
      output_datasets.append(labels[o].batch(micro_batch_size).shard(num_micro_batches, m))

  end_seq_dataset = [tf.data.Dataset.from_tensors(tf.constant(0.0)).batch(1).repeat()]

  # create generator function that implements an iterator over pairs of
  # input/output dict's, which map from the input,recv_tag,send_tag/output layer name 
  # to an input,recv_tag,send_tag/output sample
  def generator():
    for inout in zip(*input_datasets, *recv_tag_datasets, *send_tag_datasets,
                     *start_seq_dataset, *output_datasets, *end_seq_dataset):
      offset = 0
      inputs = dict()
      for i in range(num_inputs):
        for m in range(num_micro_batches):
          serial_index = i * num_micro_batches + m
          inputs[input_names[serial_index]] = inout[serial_index]
      offset += num_inputs * num_micro_batches

      for i in range(num_recvs):
        for m in range(num_micro_batches):
          serial_index = i * num_micro_batches + m
          inputs[recv_tag_names[serial_index]] = inout[offset + serial_index]
      offset += num_recvs * num_micro_batches

      for i in range(num_sends):
        for m in range(num_micro_batches):
          serial_index = i * num_micro_batches + m
          inputs[send_tag_names[serial_index]] = inout[offset + serial_index]
      offset += num_sends * num_micro_batches

      inputs[start_seq_name] = inout[offset]
      offset += 1

      outputs = dict()
      for o in range(num_outputs):
        for m in range(num_micro_batches):
          serial_index = o * num_micro_batches + m
          outputs[output_names[serial_index]] = inout[offset+serial_index]
      offset += num_outputs * num_micro_batches

      outputs[end_seq_name] = inout[offset]
      yield (inputs, outputs)

  input_types = {name : input_datasets[index].element_spec.dtype
                 for index, name in enumerate(input_names)}
  input_types.update({name : recv_tag_datasets[index].element_spec.dtype
                      for index, name in enumerate(recv_tag_names)})
  input_types.update({name : send_tag_datasets[index].element_spec.dtype
                      for index, name in enumerate(send_tag_names)})
  input_types.update({start_seq_name : start_seq_dataset[0].element_spec.dtype})
  output_types = {name : output_datasets[index].element_spec.dtype
                  for index, name in enumerate(output_names)}
  output_types.update({end_seq_name : end_seq_dataset[0].element_spec.dtype})

  return tf.data.Dataset.from_generator(generator, output_types=(input_types, output_types))
