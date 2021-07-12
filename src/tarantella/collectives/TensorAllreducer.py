import GPICommLib
from tarantella.collectives import utils

import tensorflow as tf
import numpy as np

class TensorAllreducer:
  def __init__(self, inputs):
    if utils.__is_nonEmptyDict__(inputs):
      self.allreducer = dict()
      for key in inputs.keys():
        self.allreducer[key] = TensorAllreducer(inputs[key])
      return

    default_tensor_id = 0
    if utils.__is_floatOrDouble__(inputs) or utils.__is_int__(inputs) or utils.__is_tensor__(inputs):
      self.shapes = [tf.shape(inputs)]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, np.asarray(inputs))]
    elif utils.__is_nonEmptyArray__(inputs):
      self.shapes = [inputs.shape]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, inputs)]
    elif utils.__is_nonEmptyList__(inputs):
      self.shapes = [tensor.shape for tensor in inputs]
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(inputs)]
    else:
      self.raise_input_error()

    self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)


  def allreduce(self, inputs):
    if utils.__is_nonEmptyDict__(inputs):
      output_dict = dict()
      for key in inputs.keys():
        output_dict[key] = self.allreducer[key].allreduce(inputs[key])
      return output_dict

    if utils.__is_floatOrDouble__(inputs) or utils.__is_int__(inputs):
      return self.allreducer.allreduce([np.asarray(inputs)])[0][0]
    elif utils.__is_tensor__(inputs):
      outputs = self.allreducer.allreduce([np.asarray(inputs)])[0]
      outputs = outputs.reshape(self.shapes[0])
      return tf.convert_to_tensor(outputs)
    elif utils.__is_nonEmptyArray__(inputs):
      outputs = self.allreducer.allreduce([inputs])[0]
      outputs = outputs.reshape(self.shapes[0])
      return outputs
    elif utils.__is_nonEmptyList__(inputs):
      outputs = self.allreducer.allreduce(inputs)
      for i, _ in enumerate(outputs):
        outputs[i] = outputs[i].reshape(self.shapes[i])
      return outputs
    else:
      self.raise_input_error()


  def raise_input_error(self):
    raise TypeError('[Tarantella][TensorAllreducer] '
                    '`inputs` should be either a dict, list, '
                    'an `np.ndarray` object, or a float/double.')
