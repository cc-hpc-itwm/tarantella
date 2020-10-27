import numpy as np
import tensorflow as tf
import GPICommLib

import tarantella.model as model
import tarantella.optimizers as optimizers
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
from tnt_tfops import tnt_ops

import sys

global_context = None

def setup_gpus(rank, ngpus = None):
  """Checks whether there are GPUs available on the machine and assigns one
  to the current rank.

  To make sure a specific GPU will be used by the current rank, TensorFlow is 
  configured so that this particular GPU is the only one visible.
  A GPU is selected if its index within the list of available GPUs is equal to
  (rank % ngpus).
  This allocation assumes that all nodes are homogeneous and are configured with
  the same number of processes (< ngpus).
 
  Args:
    rank: int, rank of the current process
    
    ngpus: int value specifying the maximum number of GPUs per node that will 
    be used. A value of "None" implies that all GPUs available on the machine 
    should be  assigned to ranks.  
    """
  
  phys_gpus = tf.config.experimental.list_physical_devices('GPU')
  print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
  print("Num GPUs Available: ", len(phys_gpus) if phys_gpus else 0)
  if phys_gpus and len(phys_gpus) > 0:
    if ngpus:
      target_gpu = rank % ngpus
      if len(phys_gpus) < ngpus:
        sys.exit("ERROR: rank %d cannot use GPU:%d (only %d GPUs available)" 
                 % (rank, target_gpu, len(phys_gpus)))
    else:
      target_gpu = rank % len(phys_gpus)
    try:
      # memory growth has to be set only once on all availble GPUs
      if target_gpu == 0:
        for gpu in phys_gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      
      # make sure only one GPU is visible per process
      tf.config.experimental.set_visible_devices(phys_gpus[target_gpu], 'GPU')
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print("[rank ", rank, "] ", len(phys_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
      print(e)

def init(ngpus = None):
  global global_context 
  if global_context is None:
    global_context = GPICommLib.GPIContext()    
    setup_gpus(global_context.rank, ngpus)

def get_rank():
  return global_context.rank

def get_size():
  return global_context.size

def broadcast_model_weights(model, root_rank = 0):
  weights = model.get_weights()
  GPICommLib.broadcast_model_weights(global_context, weights, root_rank)
  model.set_weights(weights)

class SynchCommunicator():
  def __init__(self, global_context, _fusion_threshold_bytes = 0):
    self.context = global_context
    self.weight_to_index = dict()
    self.comm = None
    self.threshold = _fusion_threshold_bytes

  def setup_infrastructure(self, gradients_and_weights):
    """ Setup state and allocate GPI segments
    """
    # Define gradient IDs associated with each weight, indexed by the weights' names
    # Assumption: the order in which the weights are provided is deterministic
    # (based on the internal TF graph description), so that all ranks process the
    # weights in the same order
    running_grad_id = 0
    for grad, weight in gradients_and_weights:
      self.weight_to_index[weight.name] = running_grad_id
      running_grad_id += 1

    # initialize the internal `SynchCommunicator` corresponding to the provided list of gradients
    grad_info = list()
    for grad, weight in gradients_and_weights:
      grad_info.append(GPICommLib.TensorInfo(self.weight_to_index[weight.name], 
                                             int(np.prod(grad.shape)), 
                                             np.dtype(grad.dtype.as_numpy_dtype)))
    self.comm = GPICommLib.SynchDistCommunicator(global_context, grad_info, self.threshold)

  def reduce_gradients(self, gradients_and_weights):
    gradients_to_reduce = list()
    for grad, weight in gradients_and_weights:
      # add an Allreduce operation for each gradient
      grad_id = self.weight_to_index[weight.name]
      output_grad = tnt_ops.start_allreduce_op(grad, tensor_id = grad_id,
                                              tnt_synchcomm = self.comm.get_raw_ptr())
      gradients_to_reduce.append(output_grad)

    # Create barrier op in the Tensorflow graph to make sure all 
    # the Allreduce operations on gradients have started.
    # This ensures that the graph execution does not get delayed by waiting 
    # for gradients to be reduced as long as there are remaining computations 
    # in the backward pass.
    temp_gradients = tnt_ops.barrier_op(gradients_to_reduce, 
                                         Tout = [tf.float32] * len(gradients_to_reduce))

    # Add individual ops that wait for each gradient to be reduced before updating 
    # the weights.
    # These ops are executed only after the backward pass has been completed.
    reduced_gradients = list()
    for idx, (_, weight) in enumerate(gradients_and_weights):
      # gradient tensors obtained after barrier are listed in the same order 
      # as the initial `gradients_and_weights`
      gradient = temp_gradients[idx]
      grad_id = self.weight_to_index[weight.name]

      output_grad = tnt_ops.finish_allreduce_op(gradient,
                                                tensor_id = grad_id,
                                                Tout = tf.float32,
                                                tnt_synchcomm = self.comm.get_raw_ptr())
      reduced_gradients.append(output_grad)
    return reduced_gradients


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

