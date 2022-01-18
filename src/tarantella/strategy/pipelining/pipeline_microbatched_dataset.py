import tarantella.strategy.pipelining.partition_info as pinfo

import numpy as np
import tensorflow as tf
from enum import Enum

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
def create_micro_batched_dataset(samples, # list of real inputs on the partition
                                 labels, # list of real labels (outputs) on the partition 
                                 partition_info, # contains list of EndpointInfo's for real/edge inputs/outputs (in same order as above)
                                 num_micro_batches, # number of "replicas"
                                 micro_batch_size,
                                 dataset_size):
  assert isinstance(samples, list)
  assert isinstance(labels, list)

  num_real_inputs = len(partition_info.get_real_ids(pinfo.EndpointDirection.inp))
  num_real_outputs = len(partition_info.get_real_ids(pinfo.EndpointDirection.out))
  num_edge_inputs = len(partition_info.get_edge_ids(pinfo.EndpointDirection.inp))
  num_edge_outputs = len(partition_info.get_edge_ids(pinfo.EndpointDirection.out))

  assert num_real_inputs == len(samples)
  assert num_real_outputs == len(labels)

  real_input_datasets = build_microbatched_datasets_real(
    pinfo.EndpointType.inp, samples, num_micro_batches, micro_batch_size)
  edge_input_datasets = build_microbatched_datasets_edge(
    pinfo.EndpointType.inp_edge, partition_info, num_micro_batches, micro_batch_size, dataset_size)
  recv_tag_datasets = build_microbatched_datasets_tags(
    pinfo.EndpointDirection.inp, partition_info, num_micro_batches)
  send_tag_datasets = build_microbatched_datasets_tags(
    pinfo.EndpointDirection.out, partition_info, num_micro_batches)
  seq_input_dataset =  build_microbatched_datasets_seq()
  
  real_output_datasets = build_microbatched_datasets_real(
    pinfo.EndpointType.out, labels, num_micro_batches, micro_batch_size)
  edge_output_datasets = build_microbatched_datasets_edge(
    pinfo.EndpointType.out_edge, partition_info, num_micro_batches, micro_batch_size, dataset_size)
  seq_output_dataset =  build_microbatched_datasets_seq()
  
  # create generator function that implements an iterator over
  # a pair of `inputs`/`outputs` dict's
  # that map from layer names to individual samples
  def generator():
    for inout in zip(*real_input_datasets, *edge_input_datasets,
                     *recv_tag_datasets, *send_tag_datasets,
                     *seq_input_dataset,
                     *real_output_datasets, *edge_output_datasets,
                     *seq_output_dataset):

      offset = 0
      size = num_real_inputs * num_micro_batches
      real_inputs = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.inp,
                                               partition_info.pid, num_real_inputs, num_micro_batches)
      offset += size

      size = num_edge_inputs * num_micro_batches
      edge_inputs = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.inp_edge,
                                               partition_info.pid, num_edge_inputs, num_micro_batches)
      offset += size

      size = num_edge_inputs * num_micro_batches
      recv_tags = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.recv_tag,
                                             partition_info.pid, num_edge_inputs, num_micro_batches)
      offset += size

      size = num_edge_outputs * num_micro_batches
      send_tags = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.send_tag,
                                             partition_info.pid, num_edge_outputs, num_micro_batches)
      offset += size

      size = 1
      seq_input = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.seq_input,
                                             partition_info.pid)
      offset += size

      size = num_real_outputs * num_micro_batches
      real_outputs = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.out,
                                                partition_info.pid, num_real_outputs, num_micro_batches)
      offset += size

      size = num_edge_outputs * num_micro_batches
      edge_outputs = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.out_edge,
                                                partition_info.pid, num_edge_outputs, num_micro_batches)
      offset += size

      size = 1
      seq_output = map_layer_names_to_samples(inout[offset:offset+size], pinfo.EndpointType.seq_output,
                                              partition_info.pid)

      inputs = real_inputs
      inputs.update(edge_inputs)
      inputs.update(recv_tags)
      inputs.update(send_tags)
      inputs.update(seq_input)
      outputs = real_outputs
      outputs.update(edge_outputs)
      outputs.update(seq_output)

      yield (inputs, outputs)

  input_types = map_layer_names_to_tensor_types(
    real_input_datasets, pinfo.EndpointType.inp, partition_info.pid, num_real_inputs, num_micro_batches)
  input_types.update(
    map_layer_names_to_tensor_types(
      edge_input_datasets, pinfo.EndpointType.inp_edge, partition_info.pid, num_edge_inputs, num_micro_batches))
  input_types.update(
    map_layer_names_to_tensor_types(
      recv_tag_datasets, pinfo.EndpointType.recv_tag, partition_info.pid, num_edge_inputs, num_micro_batches))
  input_types.update(
    map_layer_names_to_tensor_types(
      send_tag_datasets, pinfo.EndpointType.send_tag, partition_info.pid, num_edge_outputs, num_micro_batches))
  input_types.update(
    map_layer_names_to_tensor_types(
      seq_input_dataset, pinfo.EndpointType.seq_input, partition_info.pid))

  output_types = map_layer_names_to_tensor_types(
    real_output_datasets, pinfo.EndpointType.out, partition_info.pid, num_real_outputs, num_micro_batches)
  output_types.update(
    map_layer_names_to_tensor_types(
      edge_output_datasets, pinfo.EndpointType.out_edge, partition_info.pid, num_edge_outputs, num_micro_batches))
  output_types.update(
    map_layer_names_to_tensor_types(
      seq_output_dataset, pinfo.EndpointType.seq_output, partition_info.pid))

  return tf.data.Dataset.from_generator(generator, output_types=(input_types, output_types))

def build_microbatched_datasets_real(endpoint_type, datasets, num_micro_batches, micro_batch_size):
  microbatched_datasets = list()
  for i in range(len(datasets)):
    for m in range(num_micro_batches):
      microbatched_dataset = datasets[i].batch(micro_batch_size).shard(num_micro_batches, m)
      microbatched_datasets.append(microbatched_dataset)
  return microbatched_datasets

def build_microbatched_datasets_edge(endpoint_type, partition_info, num_micro_batches, micro_batch_size, dataset_size):
  microbatched_datasets = list()
  edge_data_value = 0.0
  for edge_id, edge_info in sorted(partition_info.get_infos(endpoint_type).items()):
    dataset = tf.data.Dataset.from_tensors(
      tf.constant(edge_data_value, shape=edge_info.shape[1:], dtype=edge_info.dtype))
    # Create `num_micro_batches` mock-up datasets with a total number of `dataset_size` instead of each edge input
    num_samples = dataset_size // num_micro_batches
    dataset = dataset.repeat(num_samples).batch(micro_batch_size)
    microbatched_datasets += num_micro_batches * [dataset]
  return microbatched_datasets

def build_microbatched_datasets_tags(endpoint_direction, partition_info, num_micro_batches):
  microbatched_datasets = list()
  dtype_tags = tf.int32 
  tags_per_microbatch = 1
  for connection_id in partition_info.get_edge_ids(endpoint_direction):
    for m in range(num_micro_batches):
      tag = tf.constant([m, connection_id], dtype=dtype_tags)
      microbatched_dataset = tf.data.Dataset.from_tensors(tag).batch(tags_per_microbatch).repeat()
      microbatched_datasets.append(microbatched_dataset)
  return microbatched_datasets

def build_microbatched_datasets_seq():
  dtype_seq_input = tf.float32 
  seq_per_microbatch = 1
  seq_data_value = 0.0
  seq_data = tf.constant(seq_data_value, dtype=dtype_seq_input)
  return [tf.data.Dataset.from_tensors(seq_data).batch(seq_per_microbatch).repeat()]

def map_layer_names_to_samples(samples, endpoint_type, partition_id, num_inputs = None, num_micro_batches = None):
  if endpoint_type in [pinfo.EndpointType.seq_input, pinfo.EndpointType.seq_output]:
    name = create_name_micro_batched_layer(partition_id, endpoint_type)
    return {name : samples[0]}

  mappings = dict()
  index = 0
  for i in range(num_inputs):
    for m in range(num_micro_batches):
      name = create_name_micro_batched_layer(partition_id, endpoint_type, layer_id = i, micro_batch_id = m)
      mappings[name] = samples[index]
      index += 1
  return mappings

def map_layer_names_to_tensor_types(datasets, endpoint_type, partition_id, num_inputs = None, num_micro_batches = None):
  if endpoint_type in [pinfo.EndpointType.seq_input, pinfo.EndpointType.seq_output]:
    name = create_name_micro_batched_layer(partition_id, endpoint_type)
    return {name : datasets[0].element_spec.dtype}

  mappings = dict()
  index = 0
  for i in range(num_inputs):
    for m in range(num_micro_batches):
      name = create_name_micro_batched_layer(partition_id, endpoint_type, layer_id = i, micro_batch_id = m)
      input_elem_spec = datasets[index].element_spec
      if isinstance(input_elem_spec, (tuple, list)):
        mappings[name] = type(input_elem_spec)([elem_spec.dtype for elem_spec in list(input_elem_spec)])
      else:
        mappings[name] = input_elem_spec.dtype
      index += 1
  return mappings
