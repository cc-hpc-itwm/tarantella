import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_dataset_ops

import logging

class TarantellaParallelMapDataset(ds.UnaryDataset):
  """A `Dataset` that maps a function over the elements in its input."""
  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int32, name="num_parallel_calls")
    self._preserve_cardinality = preserve_cardinality
    self._map_func = map_func # StructuredFunctionWrapper

    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(TarantellaParallelMapDataset, self).__init__(
      input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class TarantellaMapDataset(ds.UnaryDataset):
  """A `Dataset` that maps a function over the elements in its input."""
  def __init__(self,
               input_dataset,
               map_func,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality
    self._map_func = map_func # StructuredFunctionWrapper

    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(TarantellaMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"

def gen_dataset_transformations(dataset):
  """Generate the list of transformations that has been applied to a dataset
     Returns: tuple(original dataset, list of transformations)
  """
  stack = []
  while (hasattr(dataset, '_input_dataset')):
    if isinstance(dataset, ds.BatchDataset):
        kwargs = {"batch_size": dataset._batch_size,
                  "drop_remainder": dataset._drop_remainder}    
    elif isinstance(dataset, ds.PaddedBatchDataset):
        kwargs = {"batch_size": dataset._batch_size,
                  "drop_remainder": dataset._drop_remainder,
                  "padded_shapes": dataset._padded_shapes,
                  "padding_values": dataset._padding_values}        
    elif isinstance(dataset, ds.PrefetchDataset):
        kwargs = {"buffer_size": dataset._buffer_size}            
    elif isinstance(dataset, ds.RepeatDataset)  \
      or isinstance(dataset, ds.SkipDataset)    \
      or isinstance(dataset, ds.TakeDataset):
        kwargs = {"count": dataset._count}            
    elif isinstance(dataset, ds.ShuffleDataset):
        # ShuffleDataset does not save the given seed
        # instead it has two seed properties defined as
        # `self._seed, self._seed2 = random_seed.get_seed(seed)`
        # with `get_seed` defined in `tensorflow/python/framework/random_seed.py`
        if dataset._seed2 == 0:
          # there was no seed specified by the user
          seed = None
        else:
          seed = dataset._seed2
        kwargs = {"buffer_size": dataset._buffer_size,
                  "seed" : seed,
                  "reshuffle_each_iteration" : dataset._reshuffle_each_iteration}
    elif isinstance(dataset, ds.CacheDataset):
        kwargs = {"filename": dataset._filename} 
    elif isinstance(dataset, ds.FilterDataset):
        kwargs = {"predicate": dataset._predicate}
    elif isinstance(dataset, ds.FlatMapDataset):
        kwargs = {"map_func": dataset._map_func}
    elif isinstance(dataset, ds.MapDataset):
        kwargs = {"map_func": dataset._map_func,
                  "preserve_cardinality": dataset._preserve_cardinality}
    elif isinstance(dataset, ds.ParallelMapDataset):
        kwargs = {"map_func": dataset._map_func,
                  "preserve_cardinality": dataset._preserve_cardinality,
                  "num_parallel_calls": dataset._num_parallel_calls}          
    elif isinstance(dataset, ds.InterleaveDataset):
        kwargs = {"map_func": dataset._map_func,
                  "cycle_length": dataset._cycle_length,
                  "block_length": dataset._block_length}
    elif isinstance(dataset, ds.ParallelInterleaveDataset):
        kwargs = {"map_func": dataset._map_func,
                  "cycle_length": dataset._cycle_length,
                  "block_length": dataset._block_length,
                  "num_parallel_calls": dataset._num_parallel_calls}
    elif isinstance(dataset, ds.ShardDataset):
        kwargs = {"num_shards": dataset._num_shards,
                  "index": dataset._index} 
    elif isinstance(dataset, ds.WindowDataset):
        kwargs = {"size": dataset._size,
                  "shift": dataset._shift,
                  "stride": dataset._stride,
                  "drop_remainder": dataset._drop_remainder}
    elif isinstance(dataset, ds._OptionsDataset):
        kwargs = {"options": dataset._options}
    else:
        # TODO raise error if unknown dataset
        kwargs = {}
    
    if isinstance(dataset, ds.ParallelMapDataset):
      stack.append((TarantellaParallelMapDataset, kwargs))
    elif isinstance(dataset, ds.MapDataset):
      stack.append((TarantellaMapDataset, kwargs))
    else:
      stack.append((type(dataset), kwargs))
    dataset = dataset._input_dataset
  return (dataset, list(reversed(stack)))

def get_index_last_batch_operation(dataset_transformations):
  last_batch_transf_index = None
  for index, (transf, ds_kwargs) in enumerate(reversed(dataset_transformations)):
    if transf == ds.BatchDataset:
      last_batch_transf_index = len(dataset_transformations) - index - 1
      break
  return last_batch_transf_index

def get_num_samples(dataset):
  cardinality = tf.data.experimental.cardinality(dataset)

  if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
    logging.getLogger().debug("Infinite dataset detected.")
    return tf.data.experimental.INFINITE_CARDINALITY

  if cardinality != tf.data.experimental.UNKNOWN_CARDINALITY:
    logging.getLogger().debug("Dataset size is %d" % (cardinality.numpy()))
    return cardinality.numpy()

  logging.getLogger().debug("Unknown dataset size. Counting samples...")
  dataset_size = 0
  for d in dataset:
    dataset_size += 1
  logging.getLogger().debug("Dataset size is %d" % (dataset_size))
  return dataset_size

