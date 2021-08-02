import networkx as nx

import tarantella.strategy.pipelining.connection_info as cinfo

class RankMapper:
  # Arguments:
  # - pipeline_graph: nx.MultiDiGraph with partition names as nodes and connections as edges
  # - nranks: total number of ranks
  def __init__(self, pipeline_graph, nranks):
    self.pipeline_graph = pipeline_graph
    self.nranks = nranks
    self.mapping = self._map_ranks_to_partition_ids()

  def _map_ranks_to_partition_ids(self):
    assert len(self.pipeline_graph.nodes) <= self.nranks
    mapping = dict()
    for index, partition_id in enumerate(list(self.pipeline_graph.nodes)):
      rank = index
      mapping[rank] = partition_id
    return mapping

  def get_partition_for_rank(self, rank):
    if rank in self.mapping.keys():
      return self.mapping[rank]
    raise ValueError(f"[get_partition_for_rank] No partition id is mapped to rank {rank}:")

  def get_rank_for_partition(self, partition_id):
    matching_ranks = [rank for rank, p_id in self.mapping.items() if p_id == partition_id]

    if len(matching_ranks) == 1:
      return matching_ranks[0]
    raise ValueError(f"[get_rank_for_partition] Invalid mapping for partition id {partition_id}:"
                     f"matching ranks = {matching_ranks}")

  def get_connections_for_rank(self, rank):
    # Create connections dict: { conn_id: ((rank_i, rank), size_in_bytes)}
    assert rank < self.nranks
    connection_table = dict()
    for edge, edge_info in self.pipeline_graph.edges.items():
      conn_id = self.pipeline_graph.edges[edge]['connection_id']
      rank0 = self.get_rank_for_partition(edge[0])
      rank1 = self.get_rank_for_partition(edge[1])
      if rank in [rank0, rank1]:
        size_in_bytes = edge_info['number_elements'] * edge_info['dtype'].size
        connection_table[conn_id] = cinfo.ConnectionInfo((rank0, rank1), size_in_bytes)
    return connection_table
