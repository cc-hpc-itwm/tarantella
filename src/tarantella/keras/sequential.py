import tensorflow as tf

import tarantella as tnt
from tarantella import logger

class Sequential:
  @classmethod
  def from_config(cls, config, **kwargs):
    try:
      keras_model = tf.keras.Sequential.from_config(config, **kwargs)
      logger.info("Loaded model from `keras.Sequential`.")
    except:
      raise RuntimeError("""[tnt.keras.Sequential.from_config] Cannot load
            model; provided configuration is not a `keras.Sequential` model.""")
    return tnt.Model(keras_model)
