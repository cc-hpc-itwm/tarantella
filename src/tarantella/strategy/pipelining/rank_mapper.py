import networkx as nx

class RankMapper:
  # Arguments:
  # - partition_graph: nx.MultiDiGraph with partition names as nodes and connections as edges
  def __init__(self, partition_graph, nranks):
    self.partition_graph = partition_graph
    self.nranks = nranks
    assert len(self.partition_graph.nodes) >= self.nranks, \
           f"[RankMapper] The number of partitions {len(self.partition_graph.nodes)}" \
           f" is smaller than the number of ranks {nranks}."

  def get_partition_for_rank(self, rank):
    assert rank < self.nranks
    return list(self.partition_graph.nodes)[rank]

  def get_connections_for_rank(self, rank):
    # Connections dict { conn_id: (p_i, p_j)}
    assert rank < self.nranks
    connections = dict()
    for edge in self.partition_graph.edges(rank):
      conn_id = self.partition_graph.edges[edge]['connection_id']
      connections[conn_id] = edge
    return connections