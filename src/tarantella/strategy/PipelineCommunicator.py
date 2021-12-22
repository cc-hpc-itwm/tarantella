import tarantella as tnt

from tnt_tfops import tnt_ops
import GPICommLib

# Filter out the local partition-specific connection infos
# into the format required by the `GPICommLib.PipelineCommunicator`
#
# Example model graph: 
# (rank) --(connection_id)-- (rank)
# (0) --0-- (4) --4-- (6)
# (1) --1-- (4)
# (2) --2-- (5) --5-- (7)
# (3) --3-- (5)
#
# Connection table for rank 4:
# connection_table = { 0 : ((0, 4), size_of_edge_0_to_4,
#                      1 : ((1, 4), size_of_edge_0_to_4,
#                      4 : ((4, 6), size_of_edge_4_to_6 }
# table = { conn_id : tnt.strategy.pipelining.ConnectionInfo((rank_left, rank_right),
#                                                             size_per_sample_in_bytes), ... }
#           repeat for each edge and DP replica
# List of edges for rank 4:
# local_edges = { 0: (0, size_of_edge_0_to_4),
#                 1: (1, size_of_edge_0_to_4) }
# Local edges (per rank):
#         { ConnectionID : (PartnerRank, MessageSizeBytes), ... }
def extract_local_edges(connection_table, rank, micro_batch_size):
  local_edges = {}
  for connection_id, info in connection_table.items():
    if info.contains_rank(rank):
      local_edges[connection_id] = (info.get_other_rank(rank),
                                    info.get_size_in_bytes() * micro_batch_size)
  return local_edges

class PipelineCommunicator:
  def __init__(self, connection_table, micro_batch_size, num_micro_batches):
    rank = tnt.get_rank()
    self.local_edge_list = extract_local_edges(connection_table, rank, micro_batch_size)
    self.pipeline_comm = GPICommLib.PipelineCommunicator(self.local_edge_list, num_micro_batches)

  def send(self, input, connection_id, micro_batch_id):
    return tnt_ops.send_op(input,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def recv(self, input, connection_id, micro_batch_id):
    return tnt_ops.recv_op(input,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def get_local_connection_ids(self):
    return self.local_edge_list.keys()

