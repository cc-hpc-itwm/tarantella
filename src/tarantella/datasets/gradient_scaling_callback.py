import tensorflow as tf
from tensorflow.python.keras import backend as K

import tarantella as tnt
from tarantella import logger
import tarantella.datasets.dataset_helpers as helpers

def _get_scaling_factor(micro_batch_size, batch_size, num_ranks):
  return micro_batch_size * num_ranks / batch_size

def build_scaling_factor_table(rank, num_ranks, num_samples, batch_size):
  # Defines the gradient `scaling_factor` to be used for each iteration starting
  # with `start_iteration_id`
  # scaling_factor_table = { start_iteration_id: scaling_factor }

  if helpers._is_batch_multiple_num_ranks(num_ranks, batch_size) and \
     helpers._is_num_samples_multiple_batch_size(num_samples, batch_size):
    return None

  micro_batch_size = helpers._get_microbatch_size(rank, num_ranks, batch_size)

  # each iteration starting with id 0 will use a scaling factor defined by
  # the rank's micro batch size
  scaling_factor_table = { 0 : _get_scaling_factor(micro_batch_size, batch_size, num_ranks) }

  # the last iteration (with an incomplete batch) uses a separate scaling factor
  if not helpers._is_num_samples_multiple_batch_size(num_samples, batch_size):
    final_iteration_id = int(num_samples // batch_size)

    last_batch_size = helpers._get_last_incomplete_batch_size(num_samples, batch_size)
    last_micro_batch_size = helpers._get_microbatch_size(rank, num_ranks, last_batch_size)
    last_iteration_scaling_factor = _get_scaling_factor(last_micro_batch_size,
                                                        last_batch_size, num_ranks)

    scaling_factor_table[final_iteration_id] = last_iteration_scaling_factor
  return scaling_factor_table

def get_scaling_factor_by_iteration(iteration_id, scaling_factor_table):
  scaling_factor = 1.0
  for min_iteration_id, value in sorted(scaling_factor_table.items()):
    if iteration_id >= min_iteration_id:
      scaling_factor = value
  return scaling_factor

class ScalingFactorScheduler(tf.keras.callbacks.Callback):
  def __init__(self, scaling_factor_table):
    super().__init__()

    self.scaling_factor_table = dict()
    for iteration_id, scaling_factor in scaling_factor_table.items():
      self.scaling_factor_table[iteration_id] = tf.dtypes.cast(scaling_factor, tf.float32)

  def on_train_batch_begin(self, batch, logs=None):
    scaling_factor = get_scaling_factor_by_iteration(batch, self.scaling_factor_table)
    if scaling_factor != self.model.optimizer.scaling_factor:
      logger.debug(f"[Rank {tnt.get_rank()}] Setting scaling factor to {scaling_factor}")
      K.set_value(self.model.optimizer.scaling_factor, scaling_factor)


