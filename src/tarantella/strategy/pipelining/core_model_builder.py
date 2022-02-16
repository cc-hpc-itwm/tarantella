import tensorflow as tf
import tarantella.strategy.pipelining.partition_info as pinfo
from tarantella import logger

def get_digraph_endpoints(graph, endpoint_direction):
  if endpoint_direction == pinfo.EndpointDirection.inp:
    get_degree_by_direction = graph.in_degree
    real_endpoint_key = 'original_input_id'
  else:
    get_degree_by_direction = graph.out_degree
    real_endpoint_key = 'original_output_id'

  real_endpoints = dict()
  conn_endpoints = dict()
  for node in graph.get_nodes():
    if get_degree_by_direction(node.name) == 0:
      if real_endpoint_key in node.info_dict:
        real_endpoints[int(node.info_dict[real_endpoint_key])] = node.name
      else:
        conn_endpoints[int(node.info_dict['connection_id'])] = node.name

  sorted_real_endpoints = [real_endpoints[key] for key in sorted(real_endpoints.keys())]
  sorted_conn_endpoints = [conn_endpoints[key] for key in sorted(conn_endpoints.keys())]
  return sorted_real_endpoints + sorted_conn_endpoints

def formatted_inout_node(node_name):
  return [node_name, 0, 0]

def formatted_inout(node_list):
  formatted_list = []
  for node_name in node_list:
    formatted_list.append(formatted_inout_node(node_name))
  return formatted_list

def formatted_inbound_node(node_name):
  return [node_name, 0, 0, {}]

def reorder_nodes_by_info(node_list, info_key):
  indexed_nodes = dict()
  for n in node_list:
    index = int(n.info_dict[info_key])
    if index in indexed_nodes.keys():
      raise ValueError(f"[reorder_nodes_by_info] Node indices are not unique")
    indexed_nodes[index] = n
  return [indexed_nodes[index] for index in sorted(indexed_nodes.keys())]

def formatted_inbound_nodes(inbound_nodes_list):
  inbound_nodes = []
  for in_node in inbound_nodes_list:
    inbound_nodes.append(formatted_inbound_node(node_name = in_node.name))
  if len(inbound_nodes) == 0:
    return []
  return [inbound_nodes]

def build_layers_for_model_config(graph):
  layers = list()
  for node in graph.get_nodes():
    predecessors = graph.get_predecessors(node.name)
    inbound_nodes = formatted_inbound_nodes(reorder_nodes_by_info(node_list = predecessors,
                                                                  info_key = 'index'))
    node_config = {'class_name' : node.info_dict['class_name'],
                   'config' : node.info_dict['config'],
                   'inbound_nodes' : inbound_nodes,
                   'name': node.name}
    layers.append(node_config)
  return layers

def set_weights(target_model, source_model):
  for layer in target_model.layers:
    try:
      if layer.count_params() > 0:
        try:
          source_layer = source_model.get_layer(layer.name)
        except ValueError:
          raise RuntimeError(f"[CoreModelBuilder][set_weights] Cannot find layer {layer.name} "
                              "in original model.")
        layer.set_weights(source_layer.get_weights())
    except ValueError:
      pass  # layer is not built yet, so no weights are needed


class CoreModelBuilder():
  def __init__(self, model, partition_id, partition_graph):
    self.partition_id = partition_id
    self.partition_graph = partition_graph
    self.core_model = self._get_model(model)

  def _to_model_config(self, partition_id, partition_graph):
    model_config = {'layers' : build_layers_for_model_config(partition_graph),
                    'name' : partition_id,
                    'input_layers' : formatted_inout(get_digraph_endpoints(
                                                        partition_graph, pinfo.EndpointDirection.inp)),
                    'output_layers' : formatted_inout(get_digraph_endpoints(
                                                        partition_graph, pinfo.EndpointDirection.out)),
                   }
    return model_config

  def _get_model(self, model):
    logger.debug(f"Creating model for partition {self.partition_id}")
    core_model_config = self._to_model_config(self.partition_id, self.partition_graph)
    core_model = tf.keras.Model().from_config(core_model_config)

    set_weights(core_model, model)
    return core_model

  def get_model(self):
    return self.core_model
