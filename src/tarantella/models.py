import tensorflow as tf
import tarantella as tnt

from tarantella.optimizers.synchronous_distributed_optimizer import SynchDistributedOptimizer

def save_model(model, filepath, **kwargs):
  model.save(filepath, **kwargs)

def load_model(filepath, **kwargs):
  loaded_model = tf.keras.models.load_model(filepath, **kwargs)
  return tnt.Model(loaded_model)

def model_from_config(config, **kwargs):
  return tnt.Model.from_config(config)
