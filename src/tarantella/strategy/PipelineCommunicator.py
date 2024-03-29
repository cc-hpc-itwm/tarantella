import tarantella as tnt

from tnt_tfops import tnt_ops
import GPICommLib

import atexit

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
#         { ConnectionID : (PartnerRank, SizePerSampleBytes), ... }
def extract_local_edges(connection_table, rank):
  local_edges = {}
  for connection_id, info in connection_table.items():
    if info.contains_rank(rank):
      local_edges[connection_id] = (info.get_other_rank(rank),
                                    info.get_size_in_bytes())
  return local_edges

class PipelineCommunicator:
  def __init__(self, connection_table, num_micro_batches):
    atexit.register(self.close)
    rank = tnt.get_rank()
    self.local_edge_list = extract_local_edges(connection_table, rank)
    self.num_micro_batches = num_micro_batches
    self.pipeline_comm = GPICommLib.PipelineCommunicator(self.local_edge_list,
                                                         self.num_micro_batches)

  def send(self, inputs, connection_id, micro_batch_id):
    return tnt_ops.send_op(inputs,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def recv(self, inputs, connection_id, micro_batch_id):
    return tnt_ops.recv_op(inputs,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def get_local_connection_ids(self):
    return self.local_edge_list.keys()

  def setup_infrastructure(self, input_shape):
    micro_batch_size = input_shape[0] if isinstance(input_shape, list) else input_shape
    self.pipeline_comm.setup_infrastructure(micro_batch_size)

  def close(self):
    del self.pipeline_comm
