import tarantella as tnt

from tnt_tfops import tnt_ops
import GPICommLib

# Filter out a global description of partitions into
# a local list of edges (connections between two ranks)
# corresponding to the current rank.
#
# Example model graph: 
# (rank) --(connection_id)-- (rank)
# (0) --0-- (4) --4-- (6)
# (1) --1-- (4)
# (2) --2-- (5) --5-- (7)
# (3) --3-- (5)
#
# Global partition table:
# partition_table = { 0 : ((0, 4), size_of_edge_0_to_4,
#                     1 : ((1, 4), size_of_edge_0_to_4,
#                     2 : ((2, 5), size_of_edge_2_to_5,
#                     3 : ((3, 5), size_of_edge_2_to_5,
#                     4 : ((4, 6), size_of_edge_4_to_6,
#                     5 : ((5, 7), size_of_edge_5_to_7 }
# table = { conn_id : ((rank_left, rank_right), size_per_micro_batch_in_bytes), ... }
#                                                 repeat for each edge and DP replica
# Local list of edges for rank 4:
# local_edges = { 0: (0, size_of_edge_0_to_4),
#                 1: (1, size_of_edge_0_to_4) }
# Local edges (per rank):
#         { ConnectionID : (PartnerRank, MessageSizeBytes), ... }

def extract_local_edges(partition_table, rank):
  local_edges = {}
  for connection_id, info in partition_table.items():
    edge = {}
    if info[0][0] == rank:
      edge = { connection_id : (info[0][1], info[1])}
    elif info[0][1] == rank:
      edge = { connection_id : (info[0][0], info[1])}
    local_edges.update(edge)
  return local_edges

class PipelineCommunicator:
  def __init__(self, partition_table, num_micro_batches):
    rank = tnt.get_rank()
    self.local_edge_list = extract_local_edges(partition_table, rank)
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

  def send_with_acknowledgement(self, input, connection_id, micro_batch_id):
    return tnt_ops.send_with_ack_op(input,
                                    connection_id = connection_id,
                                    micro_batch_id = micro_batch_id,
                                    tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def recv_with_acknowledgement(self, input, connection_id, micro_batch_id):
    return tnt_ops.recv_with_ack_op(input,
                                    connection_id = connection_id,
                                    micro_batch_id = micro_batch_id,
                                    tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def get_local_connection_ids(self):
    return self.local_edge_list.keys()

