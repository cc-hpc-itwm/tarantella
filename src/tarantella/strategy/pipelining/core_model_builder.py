import tensorflow as tf
import tarantella.strategy.pipelining.partition_info as pinfo

def get_digraph_endpoints(graph, endpoint_direction):
  if endpoint_direction == pinfo.EndpointDirection.inp:
    all_nodes_with_degree = graph.in_degree()
    real_endpoint_key = 'original_input_id'
  else:
    all_nodes_with_degree = graph.out_degree()
    real_endpoint_key = 'original_output_id'

  endpoint_nodes = [node_name for node_name, degree in all_nodes_with_degree \
                              if degree == 0 ]
  real_endpoints = dict()
  conn_endpoints = dict()
  for node_name in endpoint_nodes:
    node_info = graph.nodes[node_name]
    if real_endpoint_key in node_info:
      real_endpoints[int(node_info[real_endpoint_key])] = node_name
    else:
      conn_endpoints[int(node_info['connection_id'])] = node_name

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

def formatted_inbound_nodes(inbound_nodes_list):
  inbound_nodes = []
  for in_node in inbound_nodes_list:
    inbound_nodes.append(formatted_inbound_node(node_name = in_node))
  if len(inbound_nodes) == 0:
    return []
  return [inbound_nodes]

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
  def __init__(self, model, partition_generator, rank_mapper, rank):
    self.partition_generator = partition_generator
    self.partition_id = rank_mapper.get_partition_for_rank(rank)
    self.core_model = self._get_model(model)

  def _to_model_config(self, partition_id, partition_graph):
    model_config = {'layers' : [],
                    'name' : partition_id,
                    'input_layers' : formatted_inout(get_digraph_endpoints(
                                                        partition_graph, pinfo.EndpointDirection.inp)),
                    'output_layers' : formatted_inout(get_digraph_endpoints(
                                                        partition_graph, pinfo.EndpointDirection.out)),
                   }

    for node_name, node_info in partition_graph.nodes.items():
      inbound_nodes = formatted_inbound_nodes(partition_graph.predecessors(node_name))
      node_config = {'class_name' : node_info['class_name'],
                     'config' : node_info['config'],
                     'inbound_nodes' : inbound_nodes,
                     'name': node_name}
      model_config['layers'] += [node_config]
    return model_config

  def _get_model(self, model):
    print(f"Creating model for partition {self.partition_id}")
    partition_graph = self.partition_generator.get_partition_graph(self.partition_id)
    core_model_config = self._to_model_config(self.partition_id, partition_graph)
    core_model = tf.keras.Model().from_config(core_model_config)

    set_weights(core_model, model)
    return core_model

  def get_model(self):
    return self.core_model
