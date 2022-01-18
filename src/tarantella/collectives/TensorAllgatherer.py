import tarantella as tnt
from tarantella.collectives import utils

import tensorflow as tf
import numpy as np
from functools import singledispatchmethod

class TensorAllgatherer:
  def __init__(self, inputs, group = tnt.Group()):
    # TensorAllgather performs a single Allgather operation
    # when the input is a scalar/array/tensor
    self.group = group
    self.algorithm = "ring"
    self.create_allgather(inputs)

  @singledispatchmethod
  def create_allgather(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    self.allgatherer = tnt.Allgatherv(group = self.group,
                                      nelems = 1,
                                      algorithm = self.algorithm,
                                      dtype = type(scalar))
  @create_allgather.register
  def _(self, tensor: np.ndarray):
    self.allgatherer = tnt.Allgatherv(group = self.group,
                                      nelems = int(np.prod(tensor.shape)),
                                      algorithm = self.algorithm,
                                      dtype = tensor.dtype)

  @create_allgather.register
  def _(self, tensor: tf.Tensor):
    self.allgatherer = tnt.Allgatherv(group = self.group,
                                      nelems = int(np.prod(tensor.shape)),
                                      algorithm = self.algorithm,
                                      dtype = tensor.numpy().dtype)

  @singledispatchmethod
  def start(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    self.allgatherer.start(scalar)

  @start.register
  def _(self, array: np.ndarray):
    self.allgatherer.start(array.flatten())

  @start.register
  def _(self, tensor: tf.Tensor):
    self.allgatherer.start(tensor.numpy().flatten())

  @singledispatchmethod
  def wait_for_completion(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    out = self.allgatherer.wait_for_completion()
    return out

  @wait_for_completion.register
  def _(self, array: np.ndarray):
    out = self.allgatherer.wait_for_completion()
    return out

  @wait_for_completion.register
  def _(self, tensor: tf.Tensor):
    out = self.allgatherer.wait_for_completion()
    return tf.convert_to_tensor(dtype=tensor.dtype)

  def allgather(self, inputs):
    self.start(inputs)
    return self.wait_for_completion(inputs)

  def _raise_input_error(self):
    raise TypeError('[Tarantella][TensorAllgather] '
                    '`inputs` should be either an `np.ndarray` object, or a float/double.')
