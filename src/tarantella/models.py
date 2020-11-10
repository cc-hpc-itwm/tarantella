import tensorflow as tf
import tarantella as tnt
from tarantella import logger

def save_model(model, filepath, **kwargs):
  if isinstance(model, tnt.Model):
    logger.info("save model from instance of tnt.Model")
  elif isinstance(model, tf.keras.Model):
    logger.info("save model from instance of tf.keras.Model")
  else:
    raise ValueError("[tnt.models.save_model] `model` needs to be either",
                     "a `tf.keras.Model`, or a `tnt.Model`")
  model.save(filepath, **kwargs)

def load_model(filepath, **kwargs):
  keras_model = tf.keras.models.load_model(filepath, **kwargs)
  # FIXME: compile tnt.Model before returning
  return tnt.Model(keras_model)

def model_from_config(config, **kwargs):
  return tnt.Model.from_config(config)

def model_from_json(json_string, **kwargs):
  keras_model = tf.keras.models.model_from_json(json_string, **kwargs)
  return tnt.Model(keras_model)

def model_from_yaml(yaml_string, **kwargs):
  keras_model = tf.keras.models.model_from_yaml(yaml_string, **kwargs)
  return tnt.Model(keras_model)

def clone_model(model, **kwargs):
  if isinstance(model, tnt.Model):
    keras_model = tf.keras.models.clone_model(model.model, **kwargs)
    logger.info("clone model from instance of tnt.Model")
  elif isinstance(model, tf.keras.Model):
    keras_model = tf.keras.models.clone_model(model, **kwargs)
    logger.info("clone model from instance of tf.keras.Model")
  else:
    raise ValueError("[tnt.models.clone_model] `model` needs to be either",
                     "a `tf.keras.Model`, or a `tnt.Model`")
  return tnt.Model(keras_model)
