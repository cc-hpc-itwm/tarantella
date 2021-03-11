from tnt_tfops import tnt_ops

class PipelineCommunicator:
  def __init__(self, partition_table, num_micro_batches):
    # TODO: create list of edges for the local partition
    self.pipeline_comm = GPICommLib.PipelineCommunicator(local_edge_list, num_micro_batches)

  def send(self, input, connection_id, micro_batch_id):
    return tnt_ops.send_op(input, 
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr())

  def recv(self, input, connection_id, micro_batch_id, output_shape):
    return tnt_ops.recv_op(input,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm.get_raw_ptr(),
                           output_shape = output_shape)