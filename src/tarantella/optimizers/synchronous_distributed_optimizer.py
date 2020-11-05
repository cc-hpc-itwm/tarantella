import tensorflow as tf
import numpy as np

import tarantella
import tarantella.optimizers.optimizer_wrapper as wrapper
from tnt_tfops import tnt_ops

class SynchDistributedOptimizer(wrapper.OptimizerWrapper):
  _HAS_AGGREGATE_GRAD = True

  def __init__(self, optimizer, name = None):
    self.optimizer = optimizer
    if name is None:
      name = "SynchDistributedOptimizer"
    super(self.__class__, self).__init__(optimizer, name = name)

    # add new attributes after the base object has been initialized
    self.comm = tarantella.SynchCommunicator(tarantella.global_context)
    self.initialized = False

  # customized gradient reduction method used by `keras.model.fit`
  # cf. https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/engine/training.py#L2696
  def _aggregate_gradients(self, grads_and_vars):
    grads_and_vars = list(grads_and_vars)

    # initialize the SynchCommunicator with gradient tensors
    if not self.initialized:
      self.comm.setup_infrastructure(grads_and_vars)
      self.initialized = True

    reduced_gradients = self.comm.reduce_gradients(grads_and_vars)
    return reduced_gradients

  # override gradient computation method used in TF2.0/2.1
  # to enable gradient reduction
  def get_gradients(self, loss, params):
    gradients_to_reduce = self.optimizer.get_gradients(loss, params)

    grads_and_vars = zip(gradients_to_reduce, params)
    reduced_gradients = self._aggregate_gradients(grads_and_vars)
    return reduced_gradients
