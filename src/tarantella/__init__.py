import numpy as np
import tensorflow as tf

import runtime.tnt_config as tnt_config
global_tnt_config = tnt_config.TarantellaConfiguration()

import logging
logger = logging.getLogger("TNT_LIB")

import runtime.logging_config as logging_config
import runtime.tf_config as tf_config
from tarantella.model import Model
import tarantella.optimizers as optimizers
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers

from tarantella import models

import sys

def setup_gpus(rank, ngpus = None):
  """Checks whether there are GPUs available on the machine and assigns one
  to the current rank.

  To make sure a specific GPU will be used by the current rank, TensorFlow is 
  configured so that this particular GPU is the only one visible.
  A GPU is selected if its index within the list of available GPUs is equal to
  (rank % ngpus).
  This allocation assumes that all nodes are homogeneous and are configured with
  the same number of processes (< ngpus).
 
  Args:
    rank: int, rank of the current process
    
    ngpus: int value specifying the maximum number of GPUs per node that will 
    be used.
    """
  if ngpus is None or ngpus <= 0:
    # Disable all GPUs
    tf.config.experimental.set_visible_devices([], 'GPU')
    visible_gpus = tf.config.experimental.get_visible_devices('GPU')
    if visible_gpus and len(visible_gpus) > 0:
      sys.exit("ERROR: [rank {}] Could not disable GPUs: {} GPUs still visible".format(
               rank, len(visible_gpus)))
  else: # try to use `ngpus` per node  
    phys_gpus = tf_config.get_available_gpus()
    if phys_gpus and len(phys_gpus) > 0:
      target_gpu = rank % ngpus
      if len(phys_gpus) < ngpus:
        sys.exit("ERROR: rank {} cannot use GPU_id={} (only {} GPUs available)".format(
                rank, target_gpu, len(phys_gpus)))

      try:
        # memory growth has to be set only once on all availble GPUs
        if target_gpu == 0:
          for gpu in phys_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # make sure only one GPU is visible per process
        tf.config.experimental.set_visible_devices(phys_gpus[target_gpu], 'GPU')
      except RuntimeError:
        raise RuntimeError("[Tarantella][init] Cannot configure GPUs")
  logger.debug("Using device: {}".format(tf.config.experimental.get_visible_devices()))

def init(devices_per_node = None):
    GPICommLib.initGaspiCxx()
    logging_config.setup_logging(logger, global_tnt_config.log_level,
                                 get_rank(), is_master_rank(),
                                 global_tnt_config.log_on_all_devices)

    # configure GPUs if a number of GPUs per node is specified, either as a parameter
    # or as a `TNT_GPUS_PER_NODE` environment variable
    if devices_per_node is None:
      devices_per_node = global_tnt_config.gpus_per_node
    setup_gpus(get_rank(), ngpus = devices_per_node)

def get_rank():
  return GPICommLib.get_rank()

def get_master_rank():
  return 0

def is_master_rank():
  return get_rank() == get_master_rank()

def get_size():
  return GPICommLib.get_size()
