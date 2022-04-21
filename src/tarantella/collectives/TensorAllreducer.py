import tarantella as tnt
from tarantella.collectives import utils

import tensorflow as tf
import numpy as np
from functools import singledispatchmethod

class TensorAllreducer:
  def __init__(self, inputs, group = tnt.Group()):
    # TensorAllreducer handles either a single Allreduce operation
    # when the input is a scalar/array/tensor
    # or a list/dictionary of `TensorAllreducer`s, respectively
    self.group = group
    self.algorithm = "recursivedoubling"
    self.reduction_op = tnt.ReductionOp.SUM
    self.create_allreduces(inputs)

  @singledispatchmethod
  def create_allreduces(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    self.shape = ()
    self.allreducer = tnt.Allreduce(group = self.group,
                                    nelems = 1,
                                    op = self.reduction_op,
                                    algorithm = self.algorithm,
                                    dtype = type(scalar))
  @create_allreduces.register
  def _(self, tensor: np.ndarray):
    self.shape = tensor.shape
    self.allreducer = tnt.Allreduce(group = self.group,
                                    nelems = int(np.prod(tensor.shape)),
                                    op = self.reduction_op,
                                    algorithm = self.algorithm,
                                    dtype = tensor.dtype)
  @create_allreduces.register
  def _(self, tensor: tf.Tensor):
    self.shape = tensor.shape
    self.allreducer = tnt.Allreduce(group = self.group,
                                    nelems = int(np.prod(tensor.shape)),
                                    op = self.reduction_op,
                                    algorithm = self.algorithm,
                                    dtype = tensor.numpy().dtype)
  @create_allreduces.register
  def _(self, inputs: dict):
    self.allreducer = { key : TensorAllreducer(inputs[key], group = self.group) for key in inputs.keys() }

  @create_allreduces.register
  def _(self, inputs: list):
    self.allreducer = [ TensorAllreducer(element, group = self.group) for element in inputs ]


  @singledispatchmethod
  def start(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    self.allreducer.start(scalar)

  @start.register
  def _(self, array: np.ndarray):
    self.allreducer.start(array.flatten())

  @start.register
  def _(self, tensor: tf.Tensor):
    self.allreducer.start(tensor.numpy().flatten())

  @start.register
  def _(self, inputs: dict):
    for key in inputs.keys():
      self.allreducer[key].start(inputs[key])

  @start.register
  def _(self, inputs: list):
    for i, element in enumerate(inputs):
      self.allreducer[i].start(element)


  @singledispatchmethod
  def wait_for_completion(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    out = self.allreducer.wait_for_completion()
    return out[0]

  @wait_for_completion.register
  def _(self, array: np.ndarray):
    out = self.allreducer.wait_for_completion()
    return out.reshape(self.shape)

  @wait_for_completion.register
  def _(self, tensor: tf.Tensor):
    out = self.allreducer.wait_for_completion()
    return tf.convert_to_tensor(out.reshape(self.shape), dtype=tensor.dtype)

  @wait_for_completion.register
  def _(self, inputs: dict):
    outputs = dict()
    for key in inputs.keys():
      outputs[key] = self.allreducer[key].wait_for_completion(inputs[key])
    return outputs

  @wait_for_completion.register
  def _(self, inputs: list):
    outputs = list()
    for i, element in enumerate(inputs):
      outputs.append(self.allreducer[i].wait_for_completion(element))
    return outputs

  def allreduce(self, inputs):
    self.start(inputs)
    return self.wait_for_completion(inputs)

  def _raise_input_error(self):
    raise TypeError('[Tarantella][TensorAllreducer] '
                    '`inputs` should be either a dict, list, '
                    'an `np.ndarray` object, or a float/double.')
