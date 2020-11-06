import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.ops import gen_dataset_ops

class TntFilterDataset(ds.UnaryUnchangedStructureDataset):
  """A `Dataset` that filters its input according to a predicate function."""
  def __init__(self,
               input_dataset,
               predicate,
               use_legacy_function=False):
    """See `Dataset.filter()` for details."""
    self._input_dataset = input_dataset
    self._predicate = predicate # StructuredFunctionWrapper

    variant_tensor = gen_dataset_ops.filter_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._flat_structure)
    super(TntFilterDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  def _transformation_name(self):
    return "Dataset.filter()"
