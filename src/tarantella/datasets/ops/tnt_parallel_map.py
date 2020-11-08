from tensorflow.python.compat import compat
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_dataset_ops

class TntParallelMapDataset(ds.UnaryDataset):
  """A `Dataset` that maps a function over the elements in its input."""
  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               deterministic,
               use_inter_op_parallelism,
               preserve_cardinality,
               use_legacy_function=False):
    self._input_dataset = input_dataset
    self._map_func = map_func # StructuredFunctionWrapper
    self._deterministic = deterministic
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality

    if not self._deterministic == "default" or compat.forward_compatible(2020, 3, 6):
      self._num_parallel_calls = ops.convert_to_tensor(
          num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
      variant_tensor = gen_dataset_ops.parallel_map_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          self._map_func.function.captured_inputs,
          f=self._map_func.function,
          num_parallel_calls=self._num_parallel_calls,
          deterministic=self._deterministic,
          use_inter_op_parallelism=self._use_inter_op_parallelism,
          preserve_cardinality=self._preserve_cardinality,
          **self._flat_structure)
    else:
      self._num_parallel_calls = ops.convert_to_tensor(
          num_parallel_calls, dtype=dtypes.int32, name="num_parallel_calls")
      variant_tensor = gen_dataset_ops.parallel_map_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          self._map_func.function.captured_inputs,
          f=self._map_func.function,
          num_parallel_calls=self._num_parallel_calls,
          use_inter_op_parallelism=self._use_inter_op_parallelism,
          preserve_cardinality=self._preserve_cardinality,
          **self._flat_structure)
    super(TntParallelMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"
