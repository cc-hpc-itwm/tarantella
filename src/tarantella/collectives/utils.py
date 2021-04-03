import GPICommLib

import numpy as np
import tensorflow as tf

def __is_nonEmptyList__(input):
  return isinstance(input, list) and len(input) != 0

def __is_nonEmptyArray__(input):
  return isinstance(input, np.ndarray) and input.size != 0

def __is_nonEmptyDict__(input):
  return isinstance(input, dict) and len(input) != 0

def __get_dict_values__(input):
  return np.array(tuple(input.values()))

def __as_dict__(input, reduced_values):
  output = dict()
  keys = np.array(tuple(input.keys()))

  for i in range(len(keys)):
    output[keys[i]] = reduced_values[i]

  return output

def get_tensor_info(tensor_id, tensor):
  return GPICommLib.TensorInfo(tensor_id,
                               int(np.prod(tensor.shape)),
                               np.dtype(tf.dtypes.as_dtype(tensor.dtype).as_numpy_dtype()))
