import GPICommLib

from . import utils

class TensorAllreducer:
  def __init__(self, input):
    if utils.__is_nonEmptyList__(input):
      tensor_infos = [utils.get_tensor_info(tid, tensor) for tid, tensor in enumerate(input)]
    elif utils.__is_nonEmptyArray__(input):
      tensor_infos = [utils.get_tensor_info(0, input)]
    else:
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")

    self.allreducer = GPICommLib.TensorAllreducer(tensor_infos)

  def allreduce(self, input):
    if utils.__is_nonEmptyList__(input):
      return self.allreducer.allreduce(input)
    elif utils.__is_nonEmptyArray__(input):
      return self.allreducer.allreduce([input])[0]
    else:
      raise TypeError("""[Tarantella][TensorAllreducer] Input should be
                      either a list or an array object and non-empty.""")
