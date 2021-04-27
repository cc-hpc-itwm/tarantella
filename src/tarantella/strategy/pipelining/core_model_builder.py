import tensorflow as tf

def get_digraph_inputs(graph):
  node_list = list()
  for node_name, degree in graph.in_degree():
    if degree == 0: # input nodes
      node_list += [node_name]
  return node_list

def get_digraph_outputs(graph):
  node_list = list()
  for node_name, degree in graph.out_degree():
    if degree == 0: # output nodes
      node_list += [node_name]
  return node_list

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

class CoreModelBuilder():
  def __init__(self, partition_generator, rank_mapper, rank):
    self.partition_generator = partition_generator
    self.partition_id = rank_mapper.get_partition_for_rank(rank)

  def _to_model_config(self, partition_id, partition):
    model_config = {'layers' : [],
                    'name' : partition_id,
                    'input_layers' : formatted_inout(get_digraph_inputs(partition)),
                    'output_layers' : formatted_inout(get_digraph_outputs(partition)),
                    }

    for node_name, node_info in partition.nodes.items():
      inbound_nodes = formatted_inbound_nodes(partition.predecessors(node_name))      
      node_config = {'class_name' : node_info['class_name'],
                     'config' : node_info['config'],
                     'inbound_nodes' : inbound_nodes,
                     'name': node_name}
      model_config['layers'] += [node_config]
    return model_config
  
  def get_model(self):
    partition = self.partition_generator.get_partition(self.partition_id)
    core_model_config = self._to_model_config(self.partition_id, partition)
    model = tf.keras.Model().from_config(core_model_config)
    return model