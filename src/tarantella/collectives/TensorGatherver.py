import GPICommLib
from tarantella.collectives import utils
import tarantella as tnt

import tensorflow as tf
import numpy as np

class TensorGatherver:
  def __init__(self, inputs, root_rank = tnt.get_master_rank()):
    if utils.is_nonEmptyDict(inputs):
      self.gatherver = dict()
      for key in inputs.keys():
        self.gatherver[key] = TensorGatherver(inputs[key], root_rank)
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

    self.gatherver = GPICommLib.TensorGatherver(tensor_infos, root_rank)
    self.root = root_rank

  def get_output_shape(self, shape, index):
    size = self.gatherver.get_output_count(index)
    for index in range(len(shape)):
      if index == 0:
        continue
      size = int(size / shape[index])
    if isinstance(shape, tuple):
      temp = list(shape)
      temp[0] = size
      return tuple(temp)
    else:
      shape[0] = size
      return shape

  def gatherv(self, inputs):
    if utils.is_nonEmptyDict(inputs):
      output_dict = dict()
      for key in inputs.keys():
        output_dict[key] = self.gatherver[key].gatherv(inputs[key])
      return output_dict

    if utils.is_floatOrDouble(inputs):
      if  tnt.get_rank() == tnt.get_master_rank():
        return self.gatherver.gatherv([np.asarray(inputs)])[0][0]
      else:
        return self.gatherver.gatherv([np.asarray(inputs)])
    elif utils.is_tensor(inputs):
      outputs = self.gatherver.gatherv([np.asarray(inputs)])
      if  tnt.get_rank() == self.root:
        outputs = outputs[0]
        outputs = outputs.reshape(self.get_output_shape(self.shapes[0], 0))
      return tf.convert_to_tensor(outputs)
    elif utils.is_nonEmptyArray(inputs):
      outputs = self.gatherver.gatherv([inputs])
      if tnt.get_rank() == self.root:
        outputs = outputs[0]
        outputs = outputs.reshape(self.get_output_shape(self.shapes[0], 0))
      return outputs
    elif utils.is_nonEmptyList(inputs):
      outputs = self.gatherver.gatherv(inputs)
      if tnt.get_rank() == self.root:
        for i, _ in enumerate(outputs):
          outputs[i] = outputs[i].reshape(self.get_output_shape(self.shapes[i], i))
      return outputs
    else:
      self.raise_input_error()


  def raise_input_error(self):
    raise TypeError('[Tarantella][TensorGatherver] '
                    '`inputs` should be either a dict, list, '
                    'an `np.ndarray` object, or a float/double.')
