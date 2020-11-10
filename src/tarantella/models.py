import tensorflow as tf
import tarantella as tnt

from tarantella.optimizers.synchronous_distributed_optimizer import SynchDistributedOptimizer

def save_model(model, filepath, **kwargs):
  model.save(filepath, **kwargs)
