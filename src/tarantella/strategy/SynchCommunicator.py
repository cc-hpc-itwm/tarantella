import tarantella as tnt
import tarantella.utilities.tf_version as version_utils

from tnt_tfops import tnt_ops
import GPICommLib

import atexit
import numpy as np
import tensorflow as tf

def get_tensor_info(tensor_id, tensor):
  return GPICommLib.TensorInfo(tensor_id,
                               int(np.prod(tensor.shape)),
                               np.dtype(tf.dtypes.as_dtype(tensor.dtype).as_numpy_dtype()))

class SynchCommunicator:
  def __init__(self, group = None):
    self.weight_to_index = dict()
    self.group = group
    self.comm = None
    self.threshold = tnt.global_tnt_config.fusion_threshold
    atexit.register(self.close)

  def close(self):
    del self.comm

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
    self.comm = GPICommLib.SynchDistCommunicator(grad_infos, self.threshold)

  def reduce_gradients(self, gradients_and_weights):
    gradients_to_reduce = list()
    for grad, weight in gradients_and_weights:
      # add an Allreduce operation for each gradient
      grad_id = self.weight_to_index[weight.name]
      number_partial_sums = tnt.get_size()
      grad = grad / number_partial_sums
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
      if version_utils.tf_version_below_equal('2.3'):
        reduced_gradients.append(output_grad)
      else:
        reduced_gradients.append((output_grad, weight))
    return reduced_gradients
