import tarantella.strategy.pipelining.partition_info as pinfo

import argparse
import re

import tensorflow as tf
from enum import Enum

class ModelInoutTypes(Enum):
    input_real = 'i'
    output_real = 'o'
    input_edge = 'i_edge'
    output_edge = 'o_edge'
    start_seq = 'start_seq'
    end_seq = 'end_seq'
    recv_tag = 'r_tag'
    send_tag = 's_tag'

def create_name_partition(partition_id):
  return 'p_%s' % (str(partition_id))

def create_name_micro_batched_layer(partition_id, element_type, layer_id = None, micro_batch_id = None):
  """Create a name for a layer of `type` either input/output, based on the partition_id,
  layer_id, and an optional micro_batch_id
  """
  if not isinstance(element_type, pinfo.EndpointType):
    raise TypeError("[create_name_micro_batched_layer] Layer type should be an `EndpointType` object")
  if not element_type in [pinfo.EndpointType.seq_input, pinfo.EndpointType.seq_output]:
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
                                                 element_type=pinfo.EndpointType.inp,
                                                 num_micro_batches=num_micro_batches)
  recv_tag_names = create_names_micro_batched_model(partition_id=partition_id,
                                                    num_core_elements=num_recvs,
                                                    element_type=pinfo.EndpointType.recv_tag,
                                                    num_micro_batches=num_micro_batches)
  send_tag_names = create_names_micro_batched_model(partition_id=partition_id,
                                                    num_core_elements=num_sends,
                                                    element_type=pinfo.EndpointType.send_tag,
                                                    num_micro_batches=num_micro_batches)
  start_seq_name = create_name_micro_batched_layer(partition_id=partition_id,
                                                   element_type=pinfo.EndpointType.seq_input)
  output_names = create_names_micro_batched_model(partition_id=partition_id,
                                                  num_core_elements=num_outputs,
                                                  element_type=pinfo.EndpointType.out,
                                                  num_micro_batches=num_micro_batches)
  end_seq_name = create_name_micro_batched_layer(partition_id=partition_id,
                                                 element_type=pinfo.EndpointType.seq_output)

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

  start_seq_dataset = [tf.data.Dataset.from_tensors(tf.constant(0.0, dtype=tf.float32)).batch(1).repeat()]

  output_datasets = list()
  for o in range(num_outputs):
    for m in range(num_micro_batches):
      output_datasets.append(labels[o].batch(micro_batch_size).shard(num_micro_batches, m))

  end_seq_dataset = [tf.data.Dataset.from_tensors(tf.constant(0.0, dtype=tf.float32)).batch(1).repeat()]

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
