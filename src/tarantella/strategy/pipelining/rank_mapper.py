import tarantella.strategy.pipelining.connection_info as cinfo

class RankMapper:
  # Arguments:
  # - pipeline_graph: graphs.MultiDirectedGraph with
  #   partition names as nodes and connections as edges
  # - nranks: total number of ranks
  def __init__(self, pipeline_graph, nranks):
    self.pipeline_graph = pipeline_graph
    self.nranks = nranks
    self.mapping = self._map_ranks_to_partition_ids()

  def _map_ranks_to_partition_ids(self):
    assert len(self.pipeline_graph.get_nodes()) <= self.nranks
    mapping = dict()
    for index, node in enumerate(self.pipeline_graph.get_nodes()):
      rank = index
      partition_id = node.name
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
    for edge in self.pipeline_graph.get_edges():
      conn_id = edge.info_dict['connection_id']
      rank0 = self.get_rank_for_partition(edge.source_node)
      rank1 = self.get_rank_for_partition(edge.target_node)
      if rank in [rank0, rank1]:
        size_in_bytes = edge.info_dict['number_elements'] * edge.info_dict['dtype'].size
        connection_table[conn_id] = cinfo.ConnectionInfo((rank0, rank1), size_in_bytes)
    return connection_table
