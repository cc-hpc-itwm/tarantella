from tarantella.strategy.parallel_strategy import ParallelStrategy
import tensorflow as tf
import tarantella as tnt
import tarantella.utilities.tf_version as version_utils
from tarantella import logger

def save_model(model, filepath, **kwargs):
  if isinstance(model, tnt.strategy.parallel_model.ParallelModel):
    logger.info("save model from instance of tnt.Model")
  elif isinstance(model, tf.keras.Model):
    logger.info("save model from instance of tf.keras.Model")
  else:
    raise ValueError("[tnt.models.save_model] `model` needs to be either",
                     "a `tf.keras.Model`, or a `tnt.Model`")
  model.save(filepath, **kwargs)

def load_model(filepath, compile = True, **kwargs):
  logger.debug("Load model from file: {}".format(filepath))
  keras_model = tf.keras.models.load_model(filepath, compile = compile, **kwargs)
  # FIXME load models with any type of parallelization strategy
  logger.warning("Loading model with the default `data parallel` strategy.")
  tnt_model = tnt.Model(keras_model, parallel_strategy = tnt.ParallelStrategy.DATA)
  if compile:
    try:
      tnt_optimizer = tnt.optimizers.Optimizer(keras_model.optimizer,
                                               group = tnt_model.group)
      tnt_model.dist_optimizer = tnt_optimizer
      tnt_model._set_internal_optimizer(tnt_model.dist_optimizer)
      tnt_model.compiled = True
      tnt_model.done_broadcast = True

      if version_utils.tf_version_below_equal('2.1'):
        tnt_model.model._experimental_run_tf_function = False
        logger.info("Set `experimental_run_tf_function` to False.")
    except:
      logger.info("The loaded model was not pre-compiled.")
  tnt_model.barrier.execute()
  return tnt_model

def model_from_config(config, **kwargs):
  logger.debug("Load model from an existing configuration")
  return tnt.Model.from_config(config)

def model_from_json(json_string, **kwargs):
  logger.debug("Load model from json")
  keras_model = tf.keras.models.model_from_json(json_string, **kwargs)
  # FIXME load models with any type of parallelization strategy
  logger.warning("Loading model with the default `data parallel` strategy.")
  return tnt.Model(keras_model, parallel_strategy = tnt.ParallelStrategy.DATA)

def model_from_yaml(yaml_string, **kwargs):
  logger.debug("Load model from yaml")
  try:
    keras_model = tf.keras.models.model_from_yaml(yaml_string, **kwargs)
    # FIXME load models with any type of parallelization strategy
    logger.warning("Loading model with the default `data parallel` strategy.")
    return tnt.Model(keras_model, parallel_strategy = tnt.ParallelStrategy.DATA)
  except:
    raise RuntimeError("[tnt.models.model_from_yaml] Cannot load model")

def clone_model(model, **kwargs):
  if isinstance(model, tnt.strategy.parallel_model.ParallelModel):
    keras_model = tf.keras.models.clone_model(model.model, **kwargs)
    logger.info("clone model from instance of tnt.Model")
  elif isinstance(model, tf.keras.Model):
    keras_model = tf.keras.models.clone_model(model, **kwargs)
    logger.info("clone model from instance of tf.keras.Model")
  else:
    raise ValueError("[tnt.models.clone_model] `model` needs to be either",
                     "a `tf.keras.Model`, or a `tnt.Model`")
  # FIXME load models with any type of parallelization strategy
  logger.warning("Loading model with the default `data parallel` strategy.")
  return tnt.Model(keras_model, parallel_strategy = tnt.ParallelStrategy.DATA)
