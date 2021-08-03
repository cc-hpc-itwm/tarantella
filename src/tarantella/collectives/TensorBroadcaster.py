import GPICommLib

import tarantella as tnt
from . import utils

class TensorBroadcaster():
  def __init__(self, input, root_rank = tnt.get_master_rank()):
    self.root_rank = root_rank
    self.shapes = list()

    if utils.is_nonEmptyList(input):
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(input)]
      self.shapes = [array.shape for array in input]
    elif utils.is_nonEmptyArray(input):
      tensor_infos = [utils.get_tensor_info(0, input)]
      self.shapes = [input.shape]
    else:
      raise TypeError("""[Tarantella][TensorBroadcaster] Input should be
                      either a list or an `np.ndarray` object and non-empty.""")

    self.broadcaster = GPICommLib.TensorBroadcaster(tensor_infos, self.root_rank)

  def broadcast(self, input = None):
    if input is not None: # call with input on root rank
      if tnt.get_rank() != self.root_rank:
        raise RuntimeError("[Tarantella][TensorBroadcaster][broadcast] "
                           "function with input must be called on root rank.")
      if utils.is_nonEmptyList(input):
        self.broadcaster.broadcast(input)
        return input
      elif utils.is_nonEmptyArray(input):
        self.broadcaster.broadcast([input])[0]
        return input
      else:
        raise TypeError("[Tarantella][TensorBroadcaster][broadcast] "
                        "Input should be either a list or an `np.ndarray` "
                        "object and non-empty.")
    else: # non root ranks
      if tnt.get_rank() == self.root_rank:
        raise RuntimeError("[Tarantella][TensorBroadcaster][broadcast] "
                           "function without input must be called on non-root rank.")
      outputs = self.broadcaster.broadcast([])
      for i in range(len(outputs)):
        outputs[i] = outputs[i].reshape(self.shapes[i])
      if len(outputs) == 1:
        return outputs[0]
      else:
        return outputs
