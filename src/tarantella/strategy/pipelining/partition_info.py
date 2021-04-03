import tarantella.strategy.pipelining.pipeline_microbatched_dataset as dataset_utils

import enum

class EndpointType(enum.Enum):
  inp = 'real_input'
  out = 'real_output'
  inp_edge = 'edge_input'
  out_edge = 'edge_output'
  seq_input = 'seq_input'
  seq_output = 'seq_output'
  recv_tag = 'recv_tag'
  send_tag = 'send_tag'


class EndpointDirection(enum.Enum):
  inp = 'input'
  out = 'output'


# Encapsulates information related to each endpoint,
# regardless of type (in/out, real/edge)
# The `endpoint_id` is the connection id in the case of edges and the input/output id
# in the original model for real inputs/outputs
class EndpointInfo:
  def __init__(self, endpoint_id, shape, dtype):
    self.endpoint_id = endpoint_id
    self.shape = shape
    self.dtype = dtype

  def endpoint_id(self):
    return self.endpoint_id

  def shape(self):
    return self.shape

  def dtype(self):
    return self.dtype


# Endpoint: a keras tensor (either `keras.Input` or the output of a layer)
# Input/Output ID: the index of the Endpoint within the (ordered) list of inputs/outputs in a model
class PartitionInfo:
  #  real_input_infos = list(EndpointInfo)
  #  edge_input_infos = list(EndpointInfo)
  # real_output_infos = list(EndpointInfo),
  # edge_output_infos = list(EndpointInfo)
  def __init__(self, partition_id, keras_model_with_split_layers):
    # build input/output info lists based on the user keras model
    # TODO: fill based on the keras model
    self.partition_id = partition_id
    self.real_input_infos = []
    self.edge_input_infos = []
    self.real_output_infos = []
    self.edge_output_infos = []

  @property
  def pid(self):
    return self.partition_id

  def get_real_ids(self, endpoint_direction):
    if endpoint_direction == EndpointDirection.inp:
      partition_infos = self.real_input_infos
    elif endpoint_direction == EndpointDirection.out:
      partition_infos = self.real_output_infos
    else:
      partition_infos = []
    return [endpoint_info.endpoint_id for endpoint_info in partition_infos]

  def get_edge_ids(self, endpoint_direction):
    if endpoint_direction == EndpointDirection.inp:
      partition_infos = self.edge_input_infos
    elif endpoint_direction == EndpointDirection.out:
      partition_infos = self.edge_output_infos
    else:
      partition_infos = []
    return [endpoint_info.endpoint_id for endpoint_info in partition_infos]

  def get_infos(self, endpoint_type):
    if endpoint_type == EndpointType.inp:
      return self.real_input_infos
    elif endpoint_type == EndpointType.inp_edge:
      return self.edge_input_infos
    elif endpoint_type == EndpointType.out:
      return self.real_output_infos
    elif endpoint_type == EndpointType.out_edge:
      return self.edge_output_infos



