import tarantella as tnt
from tarantella.collectives import utils

import tensorflow as tf

from functools import singledispatchmethod
import numpy as np

class TensorBroadcaster():
  # Broadcast an array or a list of tensors within a `group` starting from the *local* `root_rank`
  # within the group
  # e.g. root_rank = 0 represents rank 0 within the group
  def __init__(self, inputs, root_rank = 0, group = tnt.Group()):
    self.root_rank_local = root_rank
    self.group = group
    self.input_type = type(inputs)
    self.shapes = list()
    self.create_broadcast(inputs)

  @singledispatchmethod
  def create_broadcast(self, scalar):
    if not utils.is_scalar(scalar):
      self._raise_input_error()
    self.broadcaster = tnt.Broadcast(group = self.group,
                                     nelems = 1,
                                     root = self.root_rank_local,
                                     dtype = type(scalar))
  @create_broadcast.register
  def _(self, tensor: np.ndarray):
    self.shape = tensor.shape
    self.broadcaster = tnt.Broadcast(group = self.group,
                                     nelems = int(np.prod(tensor.shape)),
                                     root = self.root_rank_local,
                                     dtype = tensor.dtype)
  @create_broadcast.register
  def _(self, tensor: tf.Tensor):
    self.shape = tensor.shape
    self.broadcaster = tnt.Broadcast(group = self.group,
                                     nelems = int(np.prod(tensor.shape)),
                                     root = self.root_rank_local,
                                     dtype = tensor.numpy().dtype)
  @create_broadcast.register
  def _(self, inputs: list):
    self.broadcaster = [ TensorBroadcaster(element,
                                           root_rank = self.root_rank_local,
                                           group = self.group) for element in inputs ]

  @singledispatchmethod
  def start(self, scalar = None):
    if scalar is None:
      if isinstance(self.broadcaster, list):
        for bcaster in self.broadcaster:
          bcaster.start(None)
      else:
        self.broadcaster.start(None)
    else:
      if not utils.is_scalar(scalar):
        self._raise_input_error()
      self.broadcaster.start(scalar)

  @start.register
  def _(self, array: np.ndarray):
    self.broadcaster.start(array.flatten())

  @start.register
  def _(self, tensor: tf.Tensor):
    self.broadcaster.start(tensor.numpy().flatten())

  @start.register
  def _(self, inputs: list):
    for i, element in enumerate(inputs):
      self.broadcaster[i].start(element)

  @singledispatchmethod
  def wait_for_completion(self, scalar):
    if scalar is None:
      return self.wait_for_completion(self.input_type() if self.input_type != np.ndarray \
                                                     else self.input_type(0))
    out = self.broadcaster.wait_for_completion()
    return out[0]

  @wait_for_completion.register
  def _(self, array: np.ndarray):
    out = self.broadcaster.wait_for_completion()
    return out.reshape(self.shape)

  @wait_for_completion.register
  def _(self, tensor: tf.Tensor):
    out = self.broadcaster.wait_for_completion()
    return tf.convert_to_tensor(out.reshape(self.shape), dtype=tensor.dtype)

  @wait_for_completion.register
  def _(self, inputs: list):
    outputs = list()
    for bcaster in self.broadcaster:
      outputs.append(bcaster.wait_for_completion(None))
    return outputs

  def broadcast(self, inputs = None):
    self.start(inputs)

    out = self.wait_for_completion(self.input_type() if self.input_type != np.ndarray \
                                                     else self.input_type(0))
    return out

  def _raise_input_error(self):
    raise TypeError('[Tarantella][TensorBroadcaster] Input should be '
                    'either a non-empty list, an `np.ndarray` object, or a scalar.')
