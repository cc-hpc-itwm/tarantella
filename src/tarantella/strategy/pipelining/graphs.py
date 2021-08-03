import collections
import networkx as nx

Edge = collections.namedtuple('Edge', 'source_node target_node info_dict',
                              defaults = [dict()])
Node = collections.namedtuple('Node', 'name info_dict',
                              defaults = [dict()])

class Graph:
  def __init__(self, graph = None):
    self.nxgraph = nx.Graph() if graph is None else nx.Graph(graph._internal_graph)

  def add_edge(self, edge):
    self.nxgraph.add_edges_from([(edge.source_node, edge.target_node, edge.info_dict)])

  def add_node(self, node):
    self.nxgraph.add_nodes_from([(node.name, node.info_dict)])

  def remove_node(self, node_name):
    self.nxgraph.remove_node(node_name)

  def contains_node(self, node_name):
    return (node_name in self.nxgraph.nodes())

  def add_graph_info(self, info_dict):
    for k,v in info_dict.items():
      self.nxgraph.graph[k] = v

  def add_info_for_node(self, node_name, info_dict):
    for k,v in info_dict.items():
      self.nxgraph.nodes[node_name][k] = v

  def get_graph_info(self):
    graph_info = dict()
    for k,v in self.nxgraph.graph.items():
      graph_info[k] = v
    return graph_info

  def get_node(self, node_name):
    node_info = self.nxgraph.nodes[node_name]
    return Node(node_name, info_dict = node_info)

  def get_edge(self, source_node, target_node):
    edge_info = self.nxgraph.edges[(source_node, target_node)]
    return Edge(source_node, target_node, info_dict = edge_info)

  def get_nodes(self):
    nodes = list()
    for node_name in sorted(self.nxgraph.nodes):
      nodes.append(Node(node_name, info_dict = self.nxgraph.nodes[node_name]))
    return nodes

  def get_edges(self):
    edges = list()
    for src_target, edge_info in self.nxgraph.edges.items():
      edges.append(Edge(src_target[0], src_target[1], edge_info))
    return edges

  def get_connected_components(self):
    nxcomponents = nx.connected_components(self.nxgraph.to_undirected(as_view = True))
    components = list()
    for nxcomponent in sorted(nxcomponents):
      c = type(self)()
      c._internal_graph = self.nxgraph.subgraph(nxcomponent)
      components.append(c)
    return components

  @property
  def _internal_graph(self):
    return self.nxgraph

  @_internal_graph.setter
  def _internal_graph(self, nxgraph):
    self.nxgraph = nxgraph


class DirectedGraph(Graph):
  def __init__(self, graph = None):
    super().__init__()
    self.nxgraph = nx.DiGraph() if graph is None else nx.DiGraph(graph._internal_graph)

  def get_predecessors(self, node_name):
    nodes = list()
    for pred_name in self.nxgraph.predecessors(node_name):
      nodes.append(self.get_node(pred_name))
    return nodes

  def get_successors(self, node_name):
    nodes = list()
    for succ_name in self.nxgraph.successors(node_name):
      nodes.append(self.get_node(succ_name))
    return nodes

  def in_degree(self, node_name):
    return self.nxgraph.in_degree(node_name)

  def out_degree(self, node_name):
    return self.nxgraph.out_degree(node_name)


class MultiDirectedGraph(DirectedGraph):
  def __init__(self, graph = None):
    super().__init__()
    self.nxgraph = nx.MultiDiGraph() if graph is None \
              else nx.MultiDiGraph(graph._internal_graph)
