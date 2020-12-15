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

def load_model(filepath,compile=True,**kwargs):
  keras_model = tf.keras.models.load_model(filepath, compile = compile,**kwargs)
  tnt_model = tnt.Model(keras_model)
  if compile:
    #The model may be saved without compile.
    try:
      tnt_optimzier = tnt.distributed_optimizers.SynchDistributedOptimizer(keras_model.optimizer)
      tnt_model.orig_optimizer = keras_model.optimizer
      tnt_model.orig_optimizer_serialized = tf.keras.optimizers.serialize(keras_model.optimizer)
      tnt_model.dist_optimizer = tnt_optimzier
      keras_model.optimizer = tnt_model._get_optimizer(tnt_model.dist_optimizer)
      tnt_model.compiled = True
      tnt_model.done_broadcast = True
    except:
      logger.info("Model is loaded without compiled")    
  return tnt_model

def model_from_config(config, **kwargs):
  return tnt.Model.from_config(config)

def model_from_json(json_string, **kwargs):
  keras_model = tf.keras.models.model_from_json(json_string, **kwargs)
  return tnt.Model(keras_model)

def model_from_yaml(yaml_string, **kwargs):
  try:
    keras_model = tf.keras.models.model_from_yaml(yaml_string, **kwargs)
    return tnt.Model(keras_model)
  except:
    raise RuntimeError("[tnt.models.model_from_yaml] Cannot load model")

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
