import GPICommLib

import numpy as np
import tensorflow as tf

def __is_nonEmptyList__(input):
  return isinstance(input, list) and len(input) != 0

def __is_nonEmptyArray__(input):
  return isinstance(input, np.ndarray) and input.size != 0

def get_tensor_info(tensor_id, tensor):
  return GPICommLib.TensorInfo(tensor_id,
                               int(np.prod(tensor.shape)),
                               np.dtype(tf.dtypes.as_dtype(tensor.dtype).as_numpy_dtype()))