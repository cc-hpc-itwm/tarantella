import tarantella.strategy.pipelining.graphs as graphs
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.keras.layers as tnt_layers

import collections
import copy
import numpy as np

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
      # or
      # inbound_nodes = [['dense1', 0, 0, {}], ['dense2', 0, 0, {}]]
      info_list = layer_info['inbound_nodes'][0]
      if not isinstance(layer_info['inbound_nodes'][0][0], list):
       info_list = layer_info['inbound_nodes']
      edges_list[layer_info['name']] = [info[0] for info in info_list]
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

def _get_partition_endpoints(model, partition_graph, endpoint_direction):
  endpoint_names = _get_endpoint_names(model, endpoint_direction)
  return [node_name for node_name in endpoint_names \
                    if partition_graph.contains_node(node_name)]

def _generate_connection_id(node_name, split_layers_list):
  for index, split_layer_name in enumerate(sorted(split_layers_list)):
    if node_name == split_layer_name:
      return index
  raise ValueError(f"[_generate_connection_id] Cannot find node `{node_name}` "
                    "in the list of SplitLayer's")


Connection = collections.namedtuple('Connection', 'source target connection_id')


class GraphPartitionGenerator:
  def __init__(self, model):
    self.model = model
    self.graph = self._build_graph(model)
    self.connections = self._get_connections_from_split_layers()
    self._replace_split_layers()

    self.partitions = self._build_partitions()
    self.pipeline_graph = self._build_pipeline_graph()

  def _build_graph(self, model):
    graph = graphs.DirectedGraph()
    for layer_info in model.get_config()['layers']:
      # layer_info = {'class_name': 'Dense',
      #               'config': { ... },
      #               'inbound_nodes': [[['conv2d_2', 0, 0, {}]]],
      #               'name': 'dense1'}
      node_attributes = {'class_name': layer_info['class_name'],
                         'config': copy.deepcopy(layer_info['config'])}
      graph.add_node(graphs.Node(name = layer_info['name'], info_dict = node_attributes))

    edges_dict = get_incoming_edges_per_layer(model)
    for node, in_list in edges_dict.items():
      for index, in_node in enumerate(in_list):
        graph.add_edge(graphs.Edge(in_node, node))
        graph.add_info_for_node(in_node, {'index': index})

    self._add_endpoint_ids_by_direction(graph, model, pinfo.EndpointDirection.inp)
    self._add_endpoint_ids_by_direction(graph, model, pinfo.EndpointDirection.out)
    return graph

  def _add_endpoint_ids_by_direction(self, graph, model, endpoint_direction):
    endpoint_names = _get_endpoint_names(model, endpoint_direction)
    field_name = 'original_input_id' if endpoint_direction == pinfo.EndpointDirection.inp \
                                     else 'original_output_id'
    for index, layer_name in enumerate(endpoint_names):
      info_dict = dict()
      info_dict[field_name] = index
      info_dict['shape'] = model.get_layer(layer_name).output.shape
      graph.add_info_for_node(layer_name, info_dict)

  def _get_split_layers(self):
    split_layers = [node.name for node in self.graph.get_nodes() \
                              if tnt_layers.SplitLayer.is_split_layer(self.model, node.name)]
    return split_layers

  def _get_connections_from_split_layers(self):
    connections = dict()
    for node_name in sorted(self._get_split_layers()):
      predecessors = list(self.graph.get_predecessors(node_name))
      assert len(predecessors) == 1, \
             "[get_connections_from_split_layers] SplitLayers should only have one input."

      successors = list(self.graph.get_successors(node_name))
      assert len(successors) == 1, \
             "[get_connections_from_split_layers] SplitLayers should only have one output."

      connections[node_name] = Connection(source = predecessors[0].name,
                                          target = successors[0].name,
                                          connection_id = _generate_connection_id(
                                                          node_name, self._get_split_layers()))
    return connections

  def _get_connection_id(self, split_layer_name):
    for conn, info in self.connections.items():
      if conn == split_layer_name:
        return info.connection_id
    raise ValueError(f"[get_connection_id] Split layer \"{split_layer_name}\""
                      " not found in the graph.")

  def _replace_split_layers(self):
    for node in self._get_split_layers():
      self._replace_split_layer(node)

  def _replace_split_layer(self, layer_name):
    assert self.graph.contains_node(layer_name), \
           f"[_replace_split_layer] Cannot find {layer_name} in the graph."
    predecessors = self.graph.get_predecessors(layer_name)
    successors = self.graph.get_successors(layer_name)
    assert len(predecessors) == 1, \
           "[_replace_split_layer] Layer to be replaced can only have one input."
    assert len(successors) == 1, \
           "[_replace_split_layer] Layer to be replaced can only have one output."

    keras_layer = self.model.get_layer(name = layer_name)
    connection_id = self._get_connection_id(layer_name)
    node_index = self.graph.get_node(layer_name).info_dict['index']
    self.graph.remove_node(layer_name)

    # the input of the SplitLayer `layer_name` will become the output of the partition before it
    output_name = layer_name + '_input'
    output_layer_config = {'class_name': 'Layer',
                          'config': {'dtype': keras_layer.output.dtype,
                                     'name': output_name},
                          'shape' : keras_layer.output.shape,
                          'connection_id': connection_id,
                          'index': node_index}
    self.graph.add_node(graphs.Node(name = output_name, info_dict = output_layer_config))
    self.graph.add_edge(graphs.Edge(source_node = predecessors[0].name, target_node = output_name))

    # the output of the SplitLayer `layer_name` will become the input of the next partition
    input_name = layer_name + '_output'
    input_layer_config = {'class_name': 'InputLayer',
                          'config': {'batch_input_shape': keras_layer.output.shape,
                                     'dtype': keras_layer.output.dtype,
                                     'name': input_name},
                          'shape' : keras_layer.output.shape,
                          'connection_id': connection_id,
                          'index': node_index}
    self.graph.add_node(graphs.Node(name = input_name, info_dict = input_layer_config))
    self.graph.add_edge(graphs.Edge(source_node = input_name, target_node = successors[0].name))

  def _build_partitions(self):
    partitions = dict()
    for index, component in enumerate(self.graph.get_connected_components()):
      name = f"{index}"
      info_dict = { 'input_layers' : _get_partition_endpoints(self.model, component,
                                                              pinfo.EndpointDirection.inp),
                    'output_layers' : _get_partition_endpoints(self.model, component,
                                                               pinfo.EndpointDirection.out)}
      component.add_graph_info(info_dict)
      partitions[name] = component
    return partitions

  def _get_partition_with_node(self, node_name):
    for p_name, p in self.partitions.items():
      if p.contains_node(node_name):
        return p_name
    raise ValueError(f"[get_partition_with_node] Cannot find node `{node_name}` in any partition.")

  def _get_connection_info(self, connection_id):
    for node in self.graph.get_nodes():
      # two nodes have the same connection ID: one input and one output
      if node.info_dict.get('connection_id', None) == connection_id and \
         node.info_dict['class_name'] == 'InputLayer':

        number_elements = np.prod(node.info_dict['config']['batch_input_shape'].as_list()[1:])
        return {'connection_id' : connection_id,
                'number_elements' : number_elements,
                'dtype' : node.info_dict['config']['dtype']}

    raise ValueError(f"[_get_connection_info] Cannot find connection ID `{connection_id}` in the graph.")

  def _build_pipeline_graph(self):
    conn_graph = graphs.MultiDirectedGraph()

    if len(self.connections) == 0: # no split layers
      if len(self.graph.get_nodes()) > 0:
        first_node = self.graph.get_nodes()[0]
        conn_graph.add_node(graphs.Node(self._get_partition_with_node(first_node.name)))
      return conn_graph

    for _, conn_info in self.connections.items():
      source_partition = self._get_partition_with_node(conn_info.source)
      target_partition = self._get_partition_with_node(conn_info.target)
      if source_partition == target_partition:
        raise RuntimeError(f"[build_pipeline_graph] Incorrectly specified `SplitLayer` between "
                           f"`{conn_info.source}` and `{conn_info.target}` layers: "
                            "both sides of the split edge belong to the same partition.")

      # vertices are automatically created in `conn_graph` when an edge is added
      conn_graph.add_edge(graphs.Edge(source_node = source_partition,
                                      target_node = target_partition,
                                      info_dict = self._get_connection_info(conn_info.connection_id)))
    return conn_graph

  def get_partition_graph(self, partition_id):
    for p_name, p in self.partitions.items():
      if p_name == partition_id:
        return p
    raise ValueError(f"[get_partition_graph] Cannot find partition `{partition_id}`.")

  def get_pipeline_graph(self):
    return self.pipeline_graph

  def get_number_partitions(self):
    return len(self.partitions)
