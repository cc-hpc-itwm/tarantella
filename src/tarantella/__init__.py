import numpy as np
import tensorflow as tf
import GPICommLib

import runtime.tnt_config as tnt_config
global_context = None
global_tnt_config = tnt_config.TarantellaConfiguration()

import logging
logger = logging.getLogger("TNT_LIB")

import runtime.logging_config as logging_config
import runtime.tf_config as tf_config
from tarantella.model import Model
import tarantella.optimizers as optimizers
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
from tnt_tfops import tnt_ops
from tarantella import models

import sys

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
    be used.
    """
  if ngpus is None or ngpus <= 0:
    # Disable all GPUs
    tf.config.experimental.set_visible_devices([], 'GPU')
    visible_gpus = tf.config.experimental.get_visible_devices('GPU')
    if visible_gpus and len(visible_gpus) > 0:
      sys.exit("ERROR: [rank {}] Could not disable GPUs: {} GPUs still visible".format(
               rank, len(visible_gpus)))
  else: # try to use `ngpus` per node  
    phys_gpus = tf_config.get_available_gpus()
    if phys_gpus and len(phys_gpus) > 0:
      target_gpu = rank % ngpus
      if len(phys_gpus) < ngpus:
        sys.exit("ERROR: rank {} cannot use GPU_id={} (only {} GPUs available)".format(
                rank, target_gpu, len(phys_gpus)))

      try:
        # memory growth has to be set only once on all availble GPUs
        if target_gpu == 0:
          for gpu in phys_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # make sure only one GPU is visible per process
        tf.config.experimental.set_visible_devices(phys_gpus[target_gpu], 'GPU')
      except RuntimeError:
        raise RuntimeError("[Tarantella][init] Cannot configure GPUs")
  logger.debug("Using device: {}".format(tf.config.experimental.get_visible_devices()))

def init(devices_per_node = None):
  global global_context
  if global_context is None:
    global_context = GPICommLib.GPIContext()

    logging_config.setup_logging(logger, global_tnt_config.log_level,
                                 get_rank(), is_master_rank(),
                                 global_tnt_config.log_on_all_devices)

    # configure GPUs if a number of GPUs per node is specified, either as a parameter
    # or as a `TNT_GPUS_PER_NODE` environment variable
    if devices_per_node is None:
      devices_per_node = global_tnt_config.gpus_per_node
    setup_gpus(global_context.rank, ngpus = devices_per_node)

def get_rank():
  return global_context.rank

def get_master_rank():
  return 0

def is_master_rank():
  return get_rank() == get_master_rank()

def get_size():
  return global_context.size

def get_tensor_info(tensor_id, tensor):
  return GPICommLib.TensorInfo(tensor_id,
                               int(np.prod(tensor.shape)), 
                               np.dtype(tf.dtypes.as_dtype(tensor.dtype).as_numpy_dtype()))

class TensorBroadcaster():
  def __init__(self, tensor_list, root_rank):
    self.context = global_context
    self.root_rank = root_rank

    tensor_infos = [get_tensor_info(tid, tensor) for tid, tensor in enumerate(tensor_list)]
    self.broadcaster = GPICommLib.TensorBroadcaster(self.context,
                                                    tensor_infos,
                                                    self.root_rank)

  def broadcast(self, tensor_list):
    self.broadcaster.broadcast(tensor_list)

class Barrier():
  def __init__(self):
    self.barrier = GPICommLib.Barrier(global_context)

  def synchronize(self):
    self.barrier.blocking_barrier_all_ranks()

class SynchCommunicator():
  def __init__(self, global_context):
    self.context = global_context
    self.weight_to_index = dict()
    self.comm = None
    self.threshold = global_tnt_config.fusion_threshold

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
    grad_infos = list()
    for grad, weight in gradients_and_weights:
      grad_infos.append(get_tensor_info(self.weight_to_index[weight.name], grad))
    self.comm = GPICommLib.SynchDistCommunicator(global_context, grad_infos, self.threshold)

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

