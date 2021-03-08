from tnt_tfops import tnt_ops

class PipelineCommunicator:
  def __init__(self, pipeline_comm):
    # TODO: initialize pipeline communicator binding
    self.pipeline_comm_ptr = pipeline_comm.get_raw_ptr()
    pass

  def send(self, input, connection_id, micro_batch_id):
    return tnt_ops.send_op(input,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm_ptr)

  def recv(self, input, connection_id, micro_batch_id, output_shape):
    return tnt_ops.recv_op(input,
                           connection_id = connection_id,
                           micro_batch_id = micro_batch_id,
                           tnt_pipeline_comm = self.pipeline_comm_ptr,
                           output_shape = output_shape)

