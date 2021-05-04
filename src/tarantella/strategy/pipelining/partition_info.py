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

  def __eq__(self, other):
    if not isinstance(other, EndpointInfo):
      return False
    return self.endpoint_id == other.endpoint_id and \
           self.shape[1:] == other.shape[1:] and \
           self.dtype == other.dtype

  def __str__(self):
    return f"Endpoint id={self.endpoint_id}, shape={self.shape}, dtype={self.dtype}"

def build_endpoint_info(partition_graph, node_name, id_field_name):
  node_info = partition_graph.nodes[node_name]
  shape = node_info.get('shape', None)
  dtype = node_info['config'].get('dtype', None)
  endpoint_id = node_info.get(id_field_name, node_name)
  return EndpointInfo(endpoint_id, shape, dtype)


# Endpoint: a keras tensor (either `keras.Input` or the output of a layer)
# Input/Output ID: the index of the Endpoint within the (ordered) list of inputs/outputs in a model
class PartitionInfo:
  #  real_input_infos = list(EndpointInfo)
  #  edge_input_infos = list(EndpointInfo)
  # real_output_infos = list(EndpointInfo),
  # edge_output_infos = list(EndpointInfo)
  def __init__(self, partition_id, partition_graph = None):
    # build input/output info lists based on the partition graph of the keras model
    self.partition_id = partition_id
    self.real_input_infos = []
    self.edge_input_infos = []
    self.real_output_infos = []
    self.edge_output_infos = []
    self._fill_in_endpoint_infos(partition_graph)

  @property
  def pid(self):
    return self.partition_id

  def _fill_in_endpoint_infos(self, partition_graph):
    if partition_graph is None:
      return
    for node_name, node_info in partition_graph.nodes.items():
      if 'connection_id' in node_info:  # edge node in the partitions graph
        endpoint_info = build_endpoint_info(partition_graph, node_name, 'connection_id')
        if partition_graph.in_degree(node_name) == 0: # input node
          assert node_info['class_name'] == 'InputLayer'
          self.edge_input_infos.append(endpoint_info)
        else: # output node (can be any node in the graph)
          self.edge_output_infos.append(endpoint_info)
      elif 'original_input_id' in node_info:
        endpoint_info = build_endpoint_info(partition_graph, node_name, 'original_input_id')
        assert partition_graph.in_degree(node_name) == 0 # input node
        assert node_info['class_name'] == 'InputLayer'
        self.real_input_infos.append(endpoint_info)
      elif 'original_output_id' in node_info:
        endpoint_info = build_endpoint_info(partition_graph, node_name, 'original_output_id')
        assert partition_graph.out_degree(node_name) == 0 # output node
        self.real_output_infos.append(endpoint_info)

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

  def __eq__(self, other):
    return self.real_input_infos == other.real_input_infos and \
           self.edge_input_infos == other.edge_input_infos and \
           self.real_output_infos == other.real_output_infos and \
           self.edge_output_infos == other.edge_output_infos

