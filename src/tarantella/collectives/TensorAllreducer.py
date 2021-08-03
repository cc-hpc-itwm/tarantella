import GPICommLib
from tarantella.collectives import utils

import tensorflow as tf
import numpy as np

class TensorAllreducer:
  def __init__(self, inputs):
    if utils.is_nonEmptyDict(inputs):
      self.allreducer = dict()
      for key in inputs.keys():
        self.allreducer[key] = TensorAllreducer(inputs[key])
      return

    default_tensor_id = 0
    if utils.is_scalar(inputs) or utils.is_tensor(inputs):
      self.shapes = [tf.shape(inputs)]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, np.asarray(inputs))]
    elif utils.is_nonEmptyArray(inputs):
      self.shapes = [inputs.shape]
      tensor_infos = [utils.get_tensor_info(default_tensor_id, inputs)]
    elif utils.is_nonEmptyList(inputs):
      if utils.is_listOfTensors(inputs) or utils.is_listOfArrays(inputs):
        self.shapes = [tensor.shape for tensor in inputs]
        tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(inputs)]
      else: # arbitrary list
        self.allreducer = list()
        for element in inputs:
          self.allreducer.append(TensorAllreducer(element))
        return
    else:
      self._raise_input_error()

    self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)


  def allreduce(self, inputs):
    if utils.is_nonEmptyDict(inputs):
      output_dict = dict()
      for key in inputs.keys():
        output_dict[key] = self.allreducer[key].allreduce(inputs[key])
      return output_dict

    if utils.is_scalar(inputs):
      return self.allreducer.allreduce([np.asarray(inputs)])[0][0]
    elif utils.is_tensor(inputs):
      outputs = self.allreducer.allreduce([np.asarray(inputs)])[0]
      outputs = outputs.reshape(self.shapes[0])
      return tf.convert_to_tensor(outputs)
    elif utils.is_nonEmptyArray(inputs):
      outputs = self.allreducer.allreduce([inputs])[0]
      outputs = outputs.reshape(self.shapes[0])
      return outputs
    elif utils.is_nonEmptyList(inputs):
      if utils.is_listOfTensors(inputs):
        inputs = [np.asarray(i) for i in inputs]
        outputs = self.allreducer.allreduce(inputs)
        for i, _ in enumerate(outputs):
          outputs[i] = outputs[i].reshape(self.shapes[i])
          outputs[i] = tf.convert_to_tensor(outputs[i])
        return outputs
      elif utils.is_listOfArrays(inputs):
        outputs = self.allreducer.allreduce(inputs)
        for i, _ in enumerate(outputs):
          outputs[i] = outputs[i].reshape(self.shapes[i])
        return outputs
      else: # arbitrary list
        output_list = list()
        for index, element in enumerate(inputs):
          output_list.append(self.allreducer[index].allreduce(element))
        return output_list
    else:
      self._raise_input_error()


  def _raise_input_error(self):
    raise TypeError('[Tarantella][TensorAllreducer] '
                    '`inputs` should be either a dict, list, '
                    'an `np.ndarray` object, or a float/double.')
