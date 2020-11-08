import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_dataset_ops

from tarantella import logger
import tarantella.datasets.ops as tnt_ops

def _get_transformation_info_batch(dataset):
  kwargs = {"batch_size": dataset._batch_size,
            "drop_remainder": dataset._drop_remainder}
  return (ds.BatchDataset, kwargs)

def _get_transformation_info_cache(dataset):
  kwargs = {"filename": dataset._filename}
  return (ds.CacheDataset, kwargs)

def _get_transformation_info_filter(dataset):
  kwargs = {"predicate": dataset._predicate}
  return (tnt_ops.TntFilterDataset, kwargs)

def _get_transformation_info_flatmap(dataset):
  kwargs = {"map_func": dataset._map_func}
  return (tnt_ops.TntFlatMapDataset, kwargs)

def _get_transformation_info_interleave(dataset):
  kwargs = {"map_func": dataset._map_func,
            "cycle_length": dataset._cycle_length,
            "block_length": dataset._block_length}
  return (tnt_ops.TntInterleaveDataset, kwargs)

def _get_transformation_info_map(dataset):
  kwargs = {"map_func": dataset._map_func,
            "use_inter_op_parallelism": dataset._use_inter_op_parallelism,
            "preserve_cardinality": dataset._preserve_cardinality}
  return (tnt_ops.TntMapDataset, kwargs)

def _get_transformation_info_paddedbatch(dataset):
  kwargs = {"batch_size": dataset._batch_size,
            "padded_shapes": dataset._padded_shapes,
            "padding_values": dataset._padding_values,
            "drop_remainder": dataset._drop_remainder}
  return (tnt_ops.TntPaddedBatchDataset, kwargs)

def _get_transformation_info_parallelinterleave(dataset):
  # bug in TF2.2: `deterministic` is not saved as an attribute
  deterministic = "true"
  if hasattr(dataset, '_deterministic'):
    deterministic = dataset._deterministic

  kwargs = {"map_func": dataset._map_func,
            "cycle_length": dataset._cycle_length,
            "block_length": dataset._block_length,
            "num_parallel_calls": dataset._num_parallel_calls,
            "buffer_output_elements": dataset._buffer_output_elements,
            "prefetch_input_elements": dataset._prefetch_input_elements,
            "deterministic": deterministic}
  return (tnt_ops.TntParallelInterleaveDataset, kwargs)

def _get_transformation_info_parallelmap(dataset):
  kwargs = {"map_func": dataset._map_func,
            "use_inter_op_parallelism": dataset._use_inter_op_parallelism,
            "num_parallel_calls": dataset._num_parallel_calls,
            "preserve_cardinality": dataset._preserve_cardinality,
            "deterministic": dataset._deterministic}
  return (tnt_ops.TntParallelMapDataset, kwargs)

def _get_transformation_info_prefetch(dataset):
  buffer_size = dataset._buffer_size
  # TF2.2: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/data/ops/dataset_ops.py#L4255
  if buffer_size == -1:
    buffer_size = None
  kwargs = {"buffer_size" : dataset._buffer_size}
  return (ds.PrefetchDataset, kwargs)

def _get_transformation_info_repeat(dataset):
  count = dataset._count
  if count == -1:
    count = None
  kwargs = {"count": count}
  return (ds.RepeatDataset, kwargs)

def _get_transformation_info_shard(dataset):
  kwargs =  {"num_shards": dataset._num_shards,
             "index": dataset._index}
  return (ds.ShardDataset, kwargs)

def _get_transformation_info_shuffle(dataset):
  # TF 2.0 - 2.2
  # ShuffleDataset does not save the given seed
  # instead it has two seed properties defined as
  # `self._seed, self._seed2 = random_seed.get_seed(seed)`
  # with `get_seed` defined in `tensorflow/python/framework/random_seed.py` [TF2.2]
  if dataset._seed2 == 0:
    # there was no seed specified by the user
    seed = None
  else:
    seed = dataset._seed2
  kwargs = {"buffer_size": dataset._buffer_size,
            "seed": seed,
            "reshuffle_each_iteration": dataset._reshuffle_each_iteration}
  return (ds.ShuffleDataset, kwargs)

def _get_transformation_info_skip(dataset):
  kwargs = {"count": dataset._count}
  return (ds.SkipDataset, kwargs)

def _get_transformation_info_take(dataset):
  kwargs = {"count": dataset._count}
  return (ds.TakeDataset, kwargs)

def _get_transformation_info_unbatch(dataset):
  kwargs = {}
  return (ds._UnbatchDataset, kwargs)

def _get_transformation_info_window(dataset):
  kwargs = {"size": dataset._size,
            "shift": dataset._shift,
            "stride": dataset._stride,
            "drop_remainder": dataset._drop_remainder}
  return (ds.WindowDataset, kwargs)

def _get_transformation_info_withoptions(dataset):
  kwargs = {"options": dataset._options}
  return (ds._OptionsDataset, kwargs)

_transformations = {ds.BatchDataset : _get_transformation_info_batch,
                    ds.CacheDataset : _get_transformation_info_cache,
                    ds.FilterDataset : _get_transformation_info_filter,
                    ds.FlatMapDataset : _get_transformation_info_flatmap,
                    ds.InterleaveDataset : _get_transformation_info_interleave,
                    ds.MapDataset : _get_transformation_info_map,
                    ds.PaddedBatchDataset : _get_transformation_info_paddedbatch,
                    ds.ParallelInterleaveDataset : _get_transformation_info_parallelinterleave,
                    ds.ParallelMapDataset : _get_transformation_info_parallelmap,
                    ds.PrefetchDataset : _get_transformation_info_prefetch,
                    ds.RepeatDataset : _get_transformation_info_repeat,
                    ds.ShardDataset : _get_transformation_info_shard,
                    ds.ShuffleDataset : _get_transformation_info_shuffle,
                    ds.SkipDataset : _get_transformation_info_skip,
                    ds.TakeDataset : _get_transformation_info_take,
                    ds._UnbatchDataset : _get_transformation_info_unbatch,
                    ds.WindowDataset : _get_transformation_info_window,
                    ds._OptionsDataset : _get_transformation_info_withoptions,  
                    }

def gen_dataset_transformations(dataset):
  """Generate the list of transformations that has been applied to a dataset
     Returns: tuple(original dataset, list of transformations)
  """
  stack = []
  while (hasattr(dataset, '_input_dataset')):
    identified_transf = False
    for transformation in _transformations:
      if isinstance(dataset, transformation):
        stack.append(_transformations[transformation](dataset))
        identified_transf = True
        break
    if not identified_transf:
      raise RuntimeError("Unknown transformation provided: {}.".format(dataset._transformation_name))
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
    logger.debug("Infinite dataset detected.")
    return tf.data.experimental.INFINITE_CARDINALITY

  if cardinality != tf.data.experimental.UNKNOWN_CARDINALITY:
    logger.debug("Dataset size is %d" % (cardinality.numpy()))
    return cardinality.numpy()

  logger.debug("Unknown dataset size. Counting samples...")
  dataset_size = 0
  for d in dataset:
    dataset_size += 1
  logger.debug("Dataset size is %d" % (dataset_size))
  return dataset_size

