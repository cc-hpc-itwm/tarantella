import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.ops import gen_dataset_ops

class TntFlatMapDataset(ds.UnaryDataset):
  """A `Dataset` that maps a function over the elements in its input and flattens the result."""
  def __init__(self,
               input_dataset,
               map_func):
    """See `Dataset.flat_map()` for details."""
    self._input_dataset = input_dataset
    self._map_func = map_func # StructuredFunctionWrapper

    variant_tensor = gen_dataset_ops.flat_map_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(TntFlatMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure._element_spec 

  def _transformation_name(self):
    return "Dataset.flat_map()"
