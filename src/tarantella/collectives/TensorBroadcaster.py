import pygpi

import tarantella as tnt
from . import utils

import numpy as np

class TensorBroadcaster():
  def __init__(self, inputs, root_rank = tnt.get_master_rank()):
    self.root_rank = root_rank
    self.shapes = list()
    self.broadcasts = list()

    if utils.is_nonEmptyArray(inputs):
      inputs = [inputs]
    elif not utils.is_nonEmptyList(inputs):
      raise TypeError("""[Tarantella][TensorBroadcaster] Input should be
                      either a list or an `np.ndarray` object and non-empty.""")
    for tensor in inputs:
      self.shapes.append(tensor.shape)
      self.broadcasts.append(pygpi.Broadcast(pygpi.Group(), int(np.prod(tensor.shape)),
                                          self.root_rank,
                                          dtype = tensor.dtype))

  def broadcast(self, inputs = None):
    outputs = list()
    for i, bcast in enumerate(self.broadcasts):
      if pygpi.get_rank() == self.root_rank:
        if utils.is_nonEmptyArray(inputs):
          inputs = [inputs]
        elif not utils.is_nonEmptyList(inputs):
          raise TypeError("""[Tarantella][TensorBroadcaster] Input should be
                          either a list or an `np.ndarray` object and non-empty.""")
        assert len(self.broadcasts) == len(inputs)

        bcast.start(inputs[i])
      else:
        bcast.start()
    for i, bcast in enumerate(self.broadcasts):
      out = bcast.wait_for_completion()
      outputs.append(out.reshape(self.shapes[i]))
    return outputs if len(outputs) > 1 else outputs[0]
    
