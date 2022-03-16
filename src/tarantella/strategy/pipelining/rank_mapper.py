import tarantella as tnt
import tarantella.strategy.pipelining.connection_info as cinfo
from tarantella import logger

class RankMapper:
  # Defines a mapping between ranks and a tuple (partition_id, replica_id)
  # Arguments:
  # - num_ranks: total number of ranks
  # - num_partitions
  def __init__(self, num_ranks, pipeline_graph):
    self.pipeline_graph = pipeline_graph
    self.num_partitions = len(self.pipeline_graph.get_nodes())
    assert self.num_partitions <= num_ranks, \
           f"[RankMapper] Number of model partitions {self.num_partitions} cannot be larger than the number of nodes."
    assert self.num_partitions > 0, "[RankMapper] Number of model partitions cannot be 0."

    self.num_ranks = num_ranks
    self.partition_mapping, self.replica_mapping = self._map_ranks_to_partition_and_replica_ids()


  def get_pipelining_group_for_rank(self, rank):
    if not rank in self.partition_mapping:
      raise ValueError(f"Rank {rank} not found in the mapping of partition IDs to ranks: {self.partition_mapping}")
    replica_id = self.get_replica_for_rank(rank)
    pipeline_group_ranks = [r for r in self.partition_mapping.keys() \
                            if self.get_replica_for_rank(r) == replica_id]

    logger.debug(f"[RankMapper] Pipeline group = {pipeline_group_ranks}.")
    return tnt.Group(pipeline_group_ranks)

  def get_replica_group_for_rank(self, rank):
    if not rank in self.replica_mapping:
      raise ValueError(f"Rank {rank} not found in the mapping of replica IDs to ranks: {self.replica_mapping}")
    partition_id = self.get_partition_for_rank(rank)
    replica_group_ranks = [r for r in self.replica_mapping.keys() \
                           if self.get_partition_for_rank(r) == partition_id]

    logger.debug(f"[RankMapper] Replica group = {replica_group_ranks}.")
    return tnt.Group(replica_group_ranks)

  def _map_ranks_to_partition_and_replica_ids(self):
    nranks_dp = self.num_ranks // self.num_partitions

    partition_mapping = dict()
    replica_mapping = dict()
    for index, node in enumerate(self.pipeline_graph.get_nodes()):
      for replica_index in range(nranks_dp):
        first_rank_in_partition = replica_index * self.num_partitions
        rank = first_rank_in_partition + index
        partition_id = node.name
        partition_mapping[rank] = partition_id
        replica_mapping[rank] = replica_index
    logger.debug(f"[RankMapper] Mapping of ranks to partition IDs: {partition_mapping}")
    logger.debug(f"[RankMapper] Mapping of ranks to replica IDs: {replica_mapping}")
    return partition_mapping, replica_mapping

  def get_partition_for_rank(self, rank):
    if rank in self.partition_mapping.keys():
      return self.partition_mapping[rank]
    raise ValueError(f"[RankMapper] No partition id is mapped to rank {rank}:")

  def get_replica_for_rank(self, rank):
    if rank in self.replica_mapping.keys():
      return self.replica_mapping[rank]
    raise ValueError(f"[RankMapper] No replica id is mapped to rank {rank}:")

  def get_connections_for_rank(self, rank):
    # Create connections dict: { conn_id: ((rank_i, rank), size_in_bytes)}
    assert rank < self.num_ranks
    partition_id = self.get_partition_for_rank(rank)
    replica_id = self.get_replica_for_rank(rank)

    connection_table = dict()
    for edge in self.pipeline_graph.get_edges():
      conn_id = edge.info_dict['connection_id']
      # `source` and `target` nodes of an edge are partition IDs
      if partition_id in [edge.source_node, edge.target_node]:
        size_in_bytes = edge.info_dict['number_elements'] * edge.info_dict['dtype'].size

        source_ranks = self._get_ranks_for_partition(edge.source_node)
        target_ranks = self._get_ranks_for_partition(edge.target_node)

        source_rank_for_replica = [r for r in source_ranks if self.get_replica_for_rank(r) == replica_id]
        target_rank_for_replica = [r for r in target_ranks if self.get_replica_for_rank(r) == replica_id]

        assert len(source_rank_for_replica) == 1, \
               f"[RankMapper] Incorrect set of ranks for replica = {replica_id}: {source_rank_for_replica}"
        assert len(target_rank_for_replica) == 1, \
               f"[RankMapper] Incorrect set of ranks for replica = {replica_id}: {target_rank_for_replica}"
        connection_table[conn_id] = cinfo.ConnectionInfo((source_rank_for_replica[0], target_rank_for_replica[0]), size_in_bytes)
    logger.debug(f"[RankMapper] Connection table for rank {rank}: {connection_table}")
    return connection_table

  def _get_ranks_for_partition(self, partition_id):
    matching_ranks = [rank for rank, p_id in self.partition_mapping.items() if p_id == partition_id]
    if len(matching_ranks) > 0:
      return matching_ranks
    raise ValueError(f"[RankMapper] Invalid mapping for partition id {partition_id}: no matching ranks")

  def _get_ranks_for_replica(self, replica_id):
    matching_ranks = [rank for rank, r_id in self.replica_mapping.items() if r_id == replica_id]
    if len(matching_ranks) > 0:
      return matching_ranks
    raise ValueError(f"[RankMapper] Invalid mapping for replica id {replica_id}: no matching ranks")
