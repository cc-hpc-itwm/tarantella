import networkx as nx

class RankMapper:
  # Arguments:
  # - partition_graph: nx.MultiDiGraph with partition names as nodes and connections as edges
  # - nranks: total number of ranks
  def __init__(self, partition_graph, nranks):
    self.partition_graph = partition_graph
    self.nranks = nranks

  def get_partition_for_rank(self, rank):
    assert rank < self.nranks
    return list(self.partition_graph.nodes)[rank]

  def get_rank_for_partition(self, partition_name):
    for index, pname in enumerate(list(self.partition_graph.nodes)):
      if pname == partition_name:
        return index # one partition per rank
    raise RuntimeError(f"[get_rank_for_partition] Unknown partition name {partition_name}")

  def get_connections_for_rank(self, rank):
    # Create connections dict: { conn_id: ((rank_i, rank), size_in_bytes)}
    assert rank < self.nranks
    partition_table = dict()
    for edge, edge_info in self.partition_graph.edges.items():
      conn_id = self.partition_graph.edges[edge]['connection_id']
      rank0 = self.get_rank_for_partition(edge[0])
      rank1 = self.get_rank_for_partition(edge[1])
      if rank in [rank0, rank1]:
        partition_table[conn_id] = ((rank0, rank1), edge_info['size'])
    return partition_table
