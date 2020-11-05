import os
import tensorflow as tf
from runtime import logger

def get_available_gpus():
  """ Checks whether there are GPUs available on the machine and assigns one
  to the current rank.
  """
  phys_gpus = tf.config.experimental.list_physical_devices('GPU')
  logger.debug("Num GPUs Available: {}".format(len(phys_gpus) if phys_gpus else 0))
  if phys_gpus is None:
    phys_gpus = []
  return len(phys_gpus)

def setup_logging(log_level):
  #tf.logging.set_verbosity(log_level) 
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
