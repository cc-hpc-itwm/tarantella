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

class GraphPartitionGenerator:
  def __init__(self, model):
    self.model = model
    self.graph = self._build_graph(model)
    self.connections = self._get_split_connections()
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

    self._add_input_ids(graph, model)
    self._add_output_ids(graph, model)
    return graph

  def _add_input_ids(self, graph, model):
    input_names = formatted_inout_to_node_names(model.get_config()['input_layers'])
    for index, layer_name in enumerate(input_names):
      graph.nodes[layer_name]['original_input_id'] = index
      graph.nodes[layer_name]['shape'] = model.get_layer(layer_name).output.shape

  def _add_output_ids(self, graph, model):
    output_names = formatted_inout_to_node_names(model.get_config()['output_layers'])
    for index, layer_name in enumerate(output_names):
      graph.nodes[layer_name]['original_output_id'] = index
      graph.nodes[layer_name]['shape'] = model.get_layer(layer_name).output.shape

  def _get_split_layers(self):
    split_layers = [node for node in self.graph.nodes() \
                         if SplitLayer.is_split_layer(self.model, node)]
    return split_layers

  def _get_split_connections(self):
    connections = dict()
    for index, node in enumerate(self._get_split_layers()):
      predecessors = list(self.graph.predecessors(node))
      assert len(predecessors) == 1, \
             "[get_split_connections] SplitLayers should only have one input."

      successors = list(self.graph.successors(node))
      assert len(successors) == 1, \
             "[get_split_connections] SplitLayers should only have one output."

      connections[node] = {'source': predecessors[0],
                           'target' : successors[0],
                           'connection_id': index
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
      self._replace_layer(node)

  def _replace_layer(self, layer_name):
    assert layer_name in self.graph.nodes(), f"[replace_layer] Cannot find {layer_name} in the graph."
    successors = list(self.graph.successors(layer_name))
    assert len(successors) == 1, "[replace_layer] Layer to be replaced can only have one output."

    keras_layer = self.model.get_layer(name = layer_name)

    connection_id = self._get_connection_id(layer_name)
    for node in self.graph.predecessors(layer_name):
      self.graph.nodes[node]['connection_id'] = connection_id
      self.graph.nodes[node]['shape'] = keras_layer.output.shape

    self.graph.remove_node(layer_name)

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
    for component in connected_components:
      name = f"{len(partitions)}"
      partitions[name] = self.graph.subgraph(component)
    return partitions

  def _get_partition_with_node(self, node_name):
    for p_name, p in self.partitions.items():
      if node_name in p.nodes:
        return p_name
    raise ValueError(f"[get_partition_with_node] Cannot find node {node_name} in any partition.")

  def _get_connection_size(self, layer_name, connection_id):
    for n in self.graph.predecessors(layer_name):
      node_info = self.graph.nodes[n]
      if node_info['connection_id'] == connection_id:
        assert node_info['class_name'] == 'InputLayer', \
               f"[get_input_size] Provided node {n} does not represent a graph input."

        size = np.prod(node_info['config']['batch_input_shape'].as_list()[1:])
        size_in_bytes = size * node_info['config']['dtype'].size
        return size_in_bytes
    raise ValueError(f"[get_input_size] Cannot find node {layer_name} in the graph.")

  def _build_partition_graph(self):
    conn_graph = nx.MultiDiGraph()

    for conn, conn_info in self.connections.items():
      source_partition = self._get_partition_with_node(conn_info['source'])
      target_partition = self._get_partition_with_node(conn_info['target'])
      connection_size_bytes = self._get_connection_size(conn_info['target'],
                                                        conn_info['connection_id'])
      conn_graph.add_edges_from([(source_partition, target_partition,
                                 {'connection_id' : conn_info['connection_id'],
                                  'size' : connection_size_bytes}) ])
    return conn_graph

  def get_partition(self, partition_id):
    for p_name, p in self.partitions.items():
      if p_name == partition_id:
        return p
    raise ValueError(f"[get_partition] Cannot find partition {input_name}.")

  def get_partition_graph(self):
    return self.partition_graph

