import tensorflow as tf

import tarantella.strategy.pipelining.partition_info as pinfo

import copy
import networkx as nx
import numpy as np


class SplitLayer(tf.keras.layers.Layer):

  def __init__(self, name='split_layer'):
    super().__init__(name=name)

  def call(self, inputs):
    return inputs

  @classmethod
  def is_split_layer(cls, model, name):
    layer = model.get_layer(name = name)
    return isinstance(layer, cls)

# TODO: add support for TF-generated Lambda layers
# Example: `x + x` as input after `SplitLayer`
# {'class_name': 'SplitLayer', 'config': {'name': 'split_layer1', 'trainable': True, 'dtype': 'float32'},
#  'name': 'split_layer1',
#  'inbound_nodes': [[['dense_relu', 0, 0, {}]]]}
# {'class_name': 'TFOpLambda', 'config': {'name': 'tf.__operators__.add', 'trainable': True,
#                                         'dtype': 'float32', 'function': '__operators__.add'},
#  'name': 'tf.__operators__.add',
#  'inbound_nodes': [['split_layer1', 0, 0, {'y': ['split_layer1', 0, 0], 'name': None}]]}
def get_incoming_edges_per_layer(model):
  edges_list = dict()
  for index, layer_info in enumerate(model.get_config()['layers']):
    if len(layer_info['inbound_nodes']) > 0:
      # inbound_nodes = [[['dense1', 0, 0, {}], ['dense2', 0, 0, {}]]]
      edges_list[layer_info['name']] = [info[0] for info in layer_info['inbound_nodes'][0]]
  return edges_list

# Extract input/output names from `keras.Model` configuration:
# { 'layers' : ...
#   'input_layers': [['input', 0, 0], ..., ['input_n', 0, 0]],
#   'output_layers': [['output', 0, 0], ..., ['output_k', 0, 0]]
# }
def formatted_inout_to_node_names(inout_list):
  node_names = list()
  for elem in inout_list:
    node_names.append(elem[0])
  return node_names

def _get_endpoint_names(model, endpoint_direction):
  field_name = 'input_layers' if endpoint_direction == pinfo.EndpointDirection.inp \
                              else 'output_layers'
  endpoint_names = formatted_inout_to_node_names(model.get_config()[field_name])
  return endpoint_names

def _get_partition_endpoints(model, partition_nodes, endpoint_direction):
  endpoint_names = _get_endpoint_names(model, endpoint_direction)
  return [name for name in endpoint_names if name in partition_nodes]

def _generate_connection_id(node_name, split_layers_list):
  for index, split_layer_name in enumerate(sorted(split_layers_list)):
    if node_name == split_layer_name:
      return index
  raise ValueError(f"[_generate_connection_id] Cannot find node `{node_name}` "
                    "in the list of SplitLayer's")


class GraphPartitionGenerator:
  def __init__(self, model):
    self.model = model
    self.graph = self._build_graph(model)
    self.connections = self._get_connections_from_split_layers()
    self._replace_split_layers()

    self.partitions = self._build_partitions()
    self.partition_graph = self._build_partition_graph()

  def _build_graph(self, model):
    graph = nx.DiGraph()
    for index, layer_info in enumerate(model.get_config()['layers']):
      # layer_info = {'class_name': 'Dense',
      #               'config': { ... },
      #               'inbound_nodes': [[['conv2d_2', 0, 0, {}]]],
      #               'name': 'dense1'}
      node_attributes = {'class_name': layer_info['class_name'],
                         'config': copy.deepcopy(layer_info['config'])}
      node_attributes['index'] = index
      graph.add_nodes_from([(layer_info['name'], node_attributes)])

    edges_list = get_incoming_edges_per_layer(model)
    for node, in_list in edges_list.items():
      for in_node in in_list:
        graph.add_edge(in_node, node)

    self._add_endpoint_ids_by_direction(graph, model, pinfo.EndpointDirection.inp)
    self._add_endpoint_ids_by_direction(graph, model, pinfo.EndpointDirection.out)
    return graph

  def _add_endpoint_ids_by_direction(self, graph, model, endpoint_direction):
    endpoint_names = _get_endpoint_names(model, endpoint_direction)
    field_name = 'original_input_id' if endpoint_direction == pinfo.EndpointDirection.inp \
                                     else 'original_output_id'
    for index, layer_name in enumerate(endpoint_names):
      graph.nodes[layer_name][field_name] = index
      graph.nodes[layer_name]['shape'] = model.get_layer(layer_name).output.shape

  def _get_split_layers(self):
    split_layers = [node for node in self.graph.nodes() \
                         if SplitLayer.is_split_layer(self.model, node)]
    return split_layers

  def _get_connections_from_split_layers(self):
    connections = dict()
    for node in sorted(self._get_split_layers()):
      predecessors = list(self.graph.predecessors(node))
      assert len(predecessors) == 1, \
             "[get_connections_from_split_layers] SplitLayers should only have one input."

      successors = list(self.graph.successors(node))
      assert len(successors) == 1, \
             "[get_connections_from_split_layers] SplitLayers should only have one output."

      connections[node] = {'source': predecessors[0],
                           'target' : successors[0],
                           'connection_id': _generate_connection_id(node, self._get_split_layers())
                          }
    return connections

  def _get_connection_id(self, split_layer_name):
    for conn, info in self.connections.items():
      if conn == split_layer_name:
        return info['connection_id']
    raise ValueError(f"[get_connection_id] Split layer \"{split_layer_name}\""
                      " not found in the graph.")

  def _replace_split_layers(self):
    for node in self._get_split_layers():
      self._replace_split_layer(node)

  def _replace_split_layer(self, layer_name):
    assert layer_name in self.graph.nodes(), f"[_replace_split_layer] Cannot find {layer_name} in the graph."
    predecessors = list(self.graph.predecessors(layer_name))
    successors = list(self.graph.successors(layer_name))
    assert len(predecessors) == 1, "[_replace_split_layer] Layer to be replaced can only have one input."
    assert len(successors) == 1, "[_replace_split_layer] Layer to be replaced can only have one output."

    keras_layer = self.model.get_layer(name = layer_name)
    connection_id = self._get_connection_id(layer_name)
    self.graph.remove_node(layer_name)

    # the input of the SplitLayer `layer_name` will become the output of the partition before it
    output_name = layer_name + '_input'
    output_layer_config = {'class_name': 'Layer',
                          'config': {'dtype': keras_layer.output.dtype,
                                     'name': output_name},
                          'shape' : keras_layer.output.shape,
                          'connection_id': connection_id}
    self.graph.add_node(output_name, **output_layer_config)
    self.graph.add_edge(predecessors[0], output_name)

    # the output of the SplitLayer `layer_name` will become the input of the next partition
    input_name = layer_name + '_output'
    input_layer_config = {'class_name': 'InputLayer',
                          'config': {'batch_input_shape': keras_layer.output.shape,
                                     'dtype': keras_layer.output.dtype,
                                     'name': input_name},
                          'shape' : keras_layer.output.shape,
                          'connection_id': connection_id}
    self.graph.add_node(input_name, **input_layer_config)
    self.graph.add_edge(input_name, successors[0])

  def _build_partitions(self):
    partitions = dict()
    connected_components = nx.connected_components(self.graph.to_undirected(as_view = True))
    for index, component in enumerate(connected_components):
      name = f"{index}"
      partitions[name] = self.graph.subgraph(component)
      partitions[name].graph['input_layers'] = _get_partition_endpoints(self.model, component, pinfo.EndpointDirection.inp)
      partitions[name].graph['output_layers'] = _get_partition_endpoints(self.model, component, pinfo.EndpointDirection.out)
    return partitions

  def _get_partition_with_node(self, node_name):
    for p_name, p in self.partitions.items():
      if node_name in p.nodes:
        return p_name
    raise ValueError(f"[get_partition_with_node] Cannot find node {node_name} in any partition.")

  def _get_connection_size(self, layer_name, connection_id):
    for n in self.graph.predecessors(layer_name):
      node_info = self.graph.nodes[n]
      if node_info.get('connection_id', None) == connection_id:
        assert node_info['class_name'] == 'InputLayer', \
               f"[get_input_size] Provided node {n} does not represent a graph input."

        size = np.prod(node_info['config']['batch_input_shape'].as_list()[1:])
        size_in_bytes = size * node_info['config']['dtype'].size
        return size_in_bytes
    raise ValueError(f"[get_input_size] Cannot find node {layer_name} in the graph.")

  def _build_partition_graph(self):
    conn_graph = nx.MultiDiGraph()

    if len(self.connections) == 0: # no split layers
      if len(self.graph.nodes) > 0:
        first_node = list(self.graph.nodes)[0]
        conn_graph.add_node(self._get_partition_with_node(first_node))
      return conn_graph

    for _, conn_info in self.connections.items():
      source_partition = self._get_partition_with_node(conn_info['source'])
      target_partition = self._get_partition_with_node(conn_info['target'])
      if source_partition == target_partition:
        raise RuntimeError(f"[build_partition_graph] Incorrectly specified `SplitLayer` between "
                           f"`{conn_info['source']}` and `{conn_info['target']}` layers: "
                            "both sides of the split edge belong to the same partition.")

      connection_size_bytes = self._get_connection_size(conn_info['target'],
                                                        conn_info['connection_id'])
      # vertices are automatically created in `conn_graph` when an edge is added
      conn_graph.add_edges_from([(source_partition, target_partition,
                                 {'connection_id' : conn_info['connection_id'],
                                  'size' : connection_size_bytes}) ])
    return conn_graph

  def get_partition(self, partition_id):
    for p_name, p in self.partitions.items():
      if p_name == partition_id:
        return p
    raise ValueError(f"[get_partition] Cannot find partition {input_name}.")

  def get_pipeline_graph(self):
    return self.partition_graph

  def get_number_partitions(self):
    return len(self.partitions)
