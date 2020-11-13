import logging
import os
import tensorflow as tf

from runtime import environment_config

def get_available_gpus():
  """ Checks whether there are GPUs available on the machine and assigns one
  to the current rank.
  """
  phys_gpus = tf.config.experimental.list_physical_devices('GPU')
  if phys_gpus is None:
    phys_gpus = []
  return phys_gpus


_tf_logging_defaults = {'TF_CPP_MIN_LOG_LEVEL' : '3',
                        }

def setup_logging(log_level, formatter_prefix, logging_filter):
  tf_env = environment_config.collect_tensorflow_variables()
  for var,value in _tf_logging_defaults.items():
    if not var in tf_env:
      os.environ[var] = value

  tf_logger = tf.get_logger()
  tf_logger.addFilter(logging_filter)
  for h in tf_logger.handlers:
    tf_logger_format = h.formatter._fmt.replace('%(levelname)s:', '').replace('%(name)s:','')
    tf_logger_format = formatter_prefix + tf_logger_format
    formatter = logging.Formatter(tf_logger_format)
    h.setFormatter(formatter)
