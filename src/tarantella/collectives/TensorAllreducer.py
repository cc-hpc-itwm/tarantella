import GPICommLib

from . import utils

import logging
import numpy as np

class TensorAllreducer:
  def __init__(self, input):
    self.shapes = list()
    tensor_infos = []

    if utils.__is_nonEmptyList__(input):
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(input)]
      self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)
      self.shapes = [array.shape for array in input]

    elif utils.__is_nonEmptyArray__(input):
      tensor_infos = [utils.get_tensor_info(len(tensor_infos), input)]
      self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)
      self.shapes = [input.shape]

    elif utils.__is_floatOrDouble__(input):
      tensor_infos = [utils.get_tensor_info(len(tensor_infos), np.asarray(input))]
      self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)

    elif utils.__is_nonEmptyDict__(input):
      self.allreducer = dict()
      for key in sorted(input.keys()):
        self.allreducer[key] = TensorAllreducer(input[key])

    else:
      logging.getLogger().info(type(input))
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")

  def allreduce(self, input):
    if utils.__is_nonEmptyList__(input):
      outputs = self.allreducer.allreduce(input)
      for i in range(len(outputs)):
        outputs[i] = outputs[i].reshape(self.shapes[i])
      return outputs

    elif utils.__is_nonEmptyArray__(input):
      outputs = self.allreducer.allreduce([input])[0]
      outputs = outputs.reshape(self.shapes[0])
      return outputs

    elif utils.__is_floatOrDouble__(input):
      return self.allreducer.allreduce([np.asarray(input)])[0][0]

    elif utils.__is_nonEmptyDict__(input):
      output_dict = dict()
      for key in sorted(input.keys()):
        output_dict[key] = self.allreducer[key].allreduce(input[key])
      return output_dict

    else:
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")
