from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_dataset_ops

class TntParallelInterleaveDataset(ds.UnaryDataset):
  """A `Dataset` that maps a function over its input and interleaves the result."""
  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length,
               num_parallel_calls,
               deterministic,
               buffer_output_elements = None,   # backward compatibility with TF2.0
               prefetch_input_elements = None):  # backward compatibility with TF2.0
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = map_func # StructuredFunctionWrapper

    self._cycle_length = cycle_length
    self._block_length = block_length
    self._buffer_output_elements = buffer_output_elements
    self._prefetch_input_elements = prefetch_input_elements
    self._num_parallel_calls = num_parallel_calls
    self._deterministic = deterministic

    if (buffer_output_elements and buffer_output_elements != ds.AUTOTUNE) or \
       (prefetch_input_elements and prefetch_input_elements != ds.AUTOTUNE):
      variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v4(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          self._map_func.function.captured_inputs,  # pylint: disable=protected-access
          self._cycle_length,
          self._block_length,
          self._buffer_output_elements,
          self._prefetch_input_elements,
          self._num_parallel_calls,
          f=self._map_func.function,
          deterministic=deterministic,
          **self._flat_structure)
    elif deterministic != "default":
      variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v3(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          self._map_func.function.captured_inputs,  # pylint: disable=protected-access
          self._cycle_length,
          self._block_length,
          self._num_parallel_calls,
          f=self._map_func.function,
          deterministic=deterministic_string,
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          self._map_func.function.captured_inputs,  # pylint: disable=protected-access
          self._cycle_length,
          self._block_length,
          self._num_parallel_calls,
          f=self._map_func.function,
          **self._flat_structure)
    super(TntParallelInterleaveDataset, self).__init__(
      input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure._element_spec

  def _transformation_name(self):
    return "Dataset.interleave()"
