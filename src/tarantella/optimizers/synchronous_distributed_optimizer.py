import tarantella as tnt
from tarantella import logger

import tensorflow as tf
import copy
from typing import Type

def _construct_from_keras_object(obj: tf.keras.optimizers.Optimizer,
                                 optimizer: tf.keras.optimizers.Optimizer) -> None:
  keras_optimizer = optimizer
  if "keras_optimizer" in keras_optimizer.__dict__.keys():
    keras_optimizer = optimizer.keras_optimizer
  for k, v in keras_optimizer.__dict__.items():
    setattr(obj, k, copy.deepcopy(v))

def _generate_default_optimizer_with_type(keras_optimizer_type: Type[tf.keras.optimizers.Optimizer]) \
                                                                    -> tf.keras.optimizers.Optimizer:
  class SynchDistributedOptimizer(keras_optimizer_type):
    _HAS_AGGREGATE_GRAD = False

    def __init__(self, keras_optimizer: tf.keras.optimizers.Optimizer,
                       name: str = None,
                       group: tnt.Group = None):
      self.keras_optimizer = keras_optimizer
      logger.debug(f"[SynchDistributedOptimizer] Initializing generic tnt.Optimizer of type={type(keras_optimizer)}")
      _construct_from_keras_object(self, keras_optimizer)

      if name is None:
        name = "SynchDistributedOptimizer"
      self.comm = tnt.SynchCommunicator(group)
      self.initialized = False
      #scaling factor to scale gradients
      self._set_hyper("scaling_factor", 1.0)
      self.gradient_aggregator = self._gradient_aggregator

    @property
    def underlying_optimizer(self):
      return self.keras_optimizer

    def _gradient_aggregator(self, grads_and_vars):
      grad,var = zip(*grads_and_vars)
      grad = list(grad)
      for i in range(len(grad)):
        grad[i] = self.scaling_factor * grad[i]
      grads_and_vars = zip(grad,var)

      grads_and_vars = list(grads_and_vars)
      # initialize the SynchCommunicator with gradient tensors
      if not self.initialized:
        self.comm.setup_infrastructure(grads_and_vars)
        self.initialized = True

      reduced_gradients = self.comm.reduce_gradients(grads_and_vars)
      return reduced_gradients

    # override gradient computation method used in TF2.0/2.1
    def get_gradients(self, loss, params):
      gradients_to_reduce = self.optimizer.get_gradients(loss, params)
      grads_and_vars = zip(gradients_to_reduce, params)
      reduced_gradients = self.gradient_aggregator(grads_and_vars)
      return reduced_gradients

  return SynchDistributedOptimizer


class OptimizerMeta(type):
  def __call__(cls, optimizer: tf.keras.optimizers.Optimizer,
                    name = None, group = None) -> tf.keras.optimizers.Optimizer:
    
    keras_optimizer_type = type(optimizer)
    TntOptimizer = _generate_default_optimizer_with_type(keras_optimizer_type)
    return TntOptimizer(optimizer,
                        name = name,
                        group = group)

class Optimizer(metaclass = OptimizerMeta):
  pass
