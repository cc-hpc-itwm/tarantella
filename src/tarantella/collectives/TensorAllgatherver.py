import GPICommLib
from tarantella.collectives import utils

import tensorflow as tf
import numpy as np

class TensorAllgatherver:
  def __init__(self, inputs):
    if utils.is_nonEmptyDict(inputs):
      self.allgatherver = dict()
      for key in inputs.keys():
        self.allgatherver[key] = TensorAllgatherver(inputs[key])
      return

    default_tensor_id = 0
    if utils.is_floatOrDouble(inputs) or utils.is_tensor(inputs):
      self.shapes = [tf.shape(inputs)]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, np.asarray(inputs))]
    elif utils.is_nonEmptyArray(inputs):
      self.shapes = [inputs.shape]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, inputs)]
    elif utils.is_nonEmptyList(inputs):
      self.shapes = [tensor.shape for tensor in inputs]
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(inputs)]
    else:
      self.raise_input_error()

    self.allgatherver = GPICommLib.TensorAllgatherver(tensor_infos)

  def get_output_shape(self, shape, index):
    size = self.allgatherver.get_output_count(index)
    if isinstance(shape, tuple):
      temp = list(shape)
      temp[0] = size
      return tuple(temp)
    else:
      shape[0] = size
      return shape

  def allgatherv(self, inputs):
    if utils.is_nonEmptyDict(inputs):
      output_dict = dict()
      for key in inputs.keys():
        output_dict[key] = self.allgatherver[key].allgatherv(inputs[key])
      return output_dict

    if utils.is_floatOrDouble(inputs):
      return self.allgatherver.allgatherv([np.asarray(inputs)])[0][0]
    elif utils.is_tensor(inputs):
      outputs = self.allgatherver.allgatherv([np.asarray(inputs)])[0]
      outputs = outputs.reshape(self.get_output_shape(self.shapes[0], 0))
      return tf.convert_to_tensor(outputs)
    elif utils.is_nonEmptyArray(inputs):
      outputs = self.allgatherver.allgatherv([inputs])[0]
      outputs = outputs.reshape(self.get_output_shape(self.shapes[0], 0))
      return outputs
    elif utils.is_nonEmptyList(inputs):
      outputs = self.allgatherver.allgatherv(inputs)
      for i, _ in enumerate(outputs):
        outputs[i] = outputs[i].reshape(self.get_output_shape(self.shapes[i], i))
      return outputs
    else:
      self.raise_input_error()


  def raise_input_error(self):
    raise TypeError('[Tarantella][TensorAllgatherver] '
                    '`inputs` should be either a dict, list, '
                    'an `np.ndarray` object, or a float/double.')
