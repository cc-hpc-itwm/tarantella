import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_dataset_ops

class TntPaddedBatchDataset(ds.UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""
  def __init__(self,
               input_dataset,
               batch_size,
               padded_shapes,
               padding_values,
               drop_remainder,
               ):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._padded_shapes = padded_shapes
    self._padding_values = padding_values
    self._drop_remainder = drop_remainder
    
    def _padded_shape_to_batch_shape(s):
      return tensor_shape.TensorShape([
          tensor_util.constant_value(self._batch_size)
          if smart_cond.smart_constant_value(self._drop_remainder) else None
      ]).concatenate(tensor_util.constant_value_as_shape(s))

    output_shapes = nest.map_structure(
        _padded_shape_to_batch_shape, self._padded_shapes)
    self._structure = structure.convert_legacy_structure(
        ds.get_legacy_output_types(self._input_dataset), output_shapes,
        ds.get_legacy_output_classes(self._input_dataset))

    variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[ ops.convert_to_tensor(s, dtype=dtypes.int64)
                          for s in nest.flatten(self._padded_shapes)
                        ],
          padding_values=nest.flatten(self._padding_values),
          drop_remainder=self._drop_remainder,
          output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(TntPaddedBatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.padded_batch()"