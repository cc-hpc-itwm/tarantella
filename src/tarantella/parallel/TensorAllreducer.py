import GPICommLib

import tarantella.parallel.utils as utils

class TensorAllreducer:
  def __init__(self, input):
    if utils.__is_nonEmptyList__(input):
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(input)]
    elif utils.__is_nonEmptyArray__(input):
      tensor_infos = [utils.get_tensor_info(0, input)]
    elif utils.__is_nonEmptyDict__(input):
      tensor_infos = [utils.get_tensor_info(0, utils.__get_dict_values__(input))]
    else:
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")

    self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)

  def allreduce(self, input):
    if utils.__is_nonEmptyList__(input):
      return self.allreducer.allreduce(input)
    elif utils.__is_nonEmptyArray__(input):
      return self.allreducer.allreduce([input])[0]
    elif utils.__is_nonEmptyDict__(input):
      return utils.__as_dict__(input,
                               self.allreducer.allreduce([utils.__get_dict_values__(input)])[0])
    else:
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")
