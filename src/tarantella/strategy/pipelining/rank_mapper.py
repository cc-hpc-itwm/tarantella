import tarantella as tnt
import tarantella.strategy.pipelining.connection_info as cinfo

class RankMapper:
  # Defines a mapping between ranks and a tuple (partition_id, replica_id)
  # Arguments:
  # - num_ranks: total number of ranks
  # - num_partitions
  def __init__(self, num_ranks, num_partitions, pipeline_graph):
    self.num_partitions = num_partitions
    self.num_ranks = num_ranks
    self.pipeline_graph = pipeline_graph
    self.mapping = self._map_ranks_to_partition_ids()

  def get_groups_for_rank(self, rank):
    replica_index = rank // self.num_partitions
    first_rank_in_partition = replica_index * self.num_partitions
    pipeline_group_ranks = range(first_rank_in_partition, first_rank_in_partition + self.num_partitions)

    nranks_dp = self.num_ranks // self.num_partitions if self.num_partitions > 1 else self.num_ranks
    first_rank_in_dp = rank - first_rank_in_partition
    dp_group_ranks = range(first_rank_in_dp, first_rank_in_dp + nranks_dp, self.num_partitions)

    return tnt.Group(pipeline_group_ranks), tnt.Group(dp_group_ranks)

  def _map_ranks_to_partition_ids(self):
    assert len(self.pipeline_graph.get_nodes()) <= self.num_ranks
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
    assert rank < self.num_ranks
    connection_table = dict()
    for edge in self.pipeline_graph.get_edges():
      conn_id = edge.info_dict['connection_id']
      rank0 = self.get_rank_for_partition(edge.source_node)
      rank1 = self.get_rank_for_partition(edge.target_node)
      if rank in [rank0, rank1]:
        size_in_bytes = edge.info_dict['number_elements'] * edge.info_dict['dtype'].size
        connection_table[conn_id] = cinfo.ConnectionInfo((rank0, rank1), size_in_bytes)
    return connection_table
