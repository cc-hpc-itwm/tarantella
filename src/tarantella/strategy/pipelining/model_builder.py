import tarantella.strategy.pipelining.pipeline_microbatched_dataset as dataset_utils
import tarantella.strategy.pipelining.partition_info as pinfo
import abc

import tensorflow as tf
import tensorflow.keras as keras

class ModelBuilder(metaclass = abc.ABCMeta):
  def __init__(self, partition_info):
    self.partition_info = partition_info
    self.shape_tags = (2,)
    self.dtype_tags = tf.int32
    self.dtype_seq_input = tf.float32

  @abc.abstractmethod
  def get_model(self):
    return None

  def split_endpoints_list(self, endpoints_list, endpoint_direction):
    if not isinstance(endpoints_list, list):
      endpoints_list = [endpoints_list]
    real_endpoints = [endpoints_list[index] \
                      for index in self.partition_info.get_real_ids(endpoint_direction)]
    edge_endpoints = [endpoints_list[index] \
                      for index in self.partition_info.get_edge_ids(endpoint_direction)]
    return real_endpoints, edge_endpoints

  def merge_endpoints(self, real_endpoints, edge_endpoints, seq_endpoint = None):
    if seq_endpoint is None:
      seq_endpoint = []
    return real_endpoints + edge_endpoints + seq_endpoint

  def merge_inputs(self, real_inputs, edge_inputs, recv_tags, send_tags, seq_input = None):
    if seq_input is None:
      seq_input = []
    inputs = self.merge_endpoints(real_inputs, edge_inputs)
    return inputs + recv_tags + send_tags + seq_input

  def build_inputs(self, endpoint_type):
    input_infos = self.partition_info.get_infos(endpoint_type)
    inputs = []
    for index, info in enumerate(input_infos):
      name = dataset_utils.create_name_micro_batched_layer(self.partition_info.pid,
                                                           element_type = endpoint_type,
                                                           layer_id = index)  # index within the real inputs
      inputs += [keras.Input(shape = info.shape[1:], dtype = info.dtype, name = name)]
    return inputs

  def build_tag_inputs(self, endpoint_direction, micro_batch_id = None):
    layer_type = pinfo.EndpointType.recv_tag if endpoint_direction==pinfo.EndpointDirection.inp \
                                                        else pinfo.EndpointType.send_tag
    tag_inputs = []
    for index in range(len(self.partition_info.get_edge_ids(endpoint_direction))):
      name = dataset_utils.create_name_micro_batched_layer(self.partition_info.pid,
                                                           element_type = layer_type,
                                                           layer_id = index,
                                                           micro_batch_id = micro_batch_id)
      tag_inputs += [keras.Input(shape=self.shape_tags, dtype=self.dtype_tags, name = name)]
    return tag_inputs

  def build_seq_input(self):
    name = dataset_utils.create_name_micro_batched_layer(self.partition_info.pid,
                                                         element_type = pinfo.EndpointType.seq_input)
    return [keras.Input(shape=(1,), dtype=self.dtype_seq_input, name = name)]

