import GPICommLib

import numpy as np
import tensorflow as tf

def is_nonEmptyList(input):
  return isinstance(input, list) and len(input) != 0

def is_nonEmptyArray(input):
  return isinstance(input, np.ndarray) and input.size != 0

def is_floatOrDouble(input):
  return isinstance(input, (np.float, np.double, np.float32, np.float64))

def is_int(input):
  return isinstance(input, (np.int, np.int16, np.int32, np.int64))

def is_scalar(input):
  return is_int(input) or is_floatOrDouble(input)

def is_tensor(input):
  return tf.is_tensor(input)

def is_nonEmptyDict(input):
  return isinstance(input, dict) and len(input) != 0

def get_tensor_info(tensor_id, tensor):
  return GPICommLib.TensorInfo(tensor_id,
                               int(np.prod(tensor.shape)),
                               np.dtype(tf.dtypes.as_dtype(tensor.dtype).as_numpy_dtype()))
