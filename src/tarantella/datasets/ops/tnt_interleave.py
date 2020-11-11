import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.ops import gen_dataset_ops

class TntInterleaveDataset(ds.UnaryDataset):
  """A `Dataset` that interleaves the result of transformed inputs."""
  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length):
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = map_func # StructuredFunctionWrapper
    self._cycle_length = cycle_length
    self._block_length = block_length
    
    variant_tensor = gen_dataset_ops.interleave_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        f=self._map_func.function,
        **self._flat_structure)
    super(TntInterleaveDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure._element_spec

  def _transformation_name(self):
    return "Dataset.interleave()"
