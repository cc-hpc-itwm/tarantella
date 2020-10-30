import logging
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.keras.engine import training_utils

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
import tarantella.datasets.distributed_dataset as ds

class TarantellaModel(tf.keras.models.Model):
  def __init__(self, model, _fusion_threshold_bytes = 32768):
    if not tarantella.global_context:
      raise RuntimeError("""Cannot initialize a TarantellaModel before the Tarantella library.
      Please call "tarantella.init() first."
      """)
      
    self.master_rank = 0
    self.rank = tarantella.get_rank()
    self.comm_size = tarantella.get_size()
    self.model = model
    self.threshold = _fusion_threshold_bytes
    self.default_shuffle_seed = 42

  def __getattr__(self, name):
    if name in ('model', 'rank', 'comm_size', 'master_rank', 'threshold'):
      return getattr(self.__dict__, name)
    return getattr(self.__dict__['model'], name)
  
  def __setattr__(self, name, value):
    if name in ('model', 'rank', 'comm_size', 'master_rank', 'threshold'):
      self.__dict__[name] = value
    else:
      setattr(self.__dict__['model'], name, value)
  
  def __delattr__(self, name):
    if name in ('model', 'rank', 'comm_size', 'master_rank', 'threshold'):
      delattr(self.__dict__, name)
    delattr(self.__dict__['model'], name)

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    
    optimizer = tarantella.distributed_optimizers.SynchDistributedOptimizer(optimizer,
                                                  _fusion_threshold_bytes = self.threshold)
    return self.model.compile(optimizer = optimizer,
                              loss = loss,
                              metrics = metrics,
                              loss_weights = loss_weights,
                              sample_weight_mode = sample_weight_mode,
                              weighted_metrics = weighted_metrics,
                              **kwargs)
                              
  def fit(self,
          x=None,
          tnt_distribute_dataset = True,
          **kwargs):

    # Broadcast initial weights to all processes
    tarantella.broadcast_model_weights(self.model, root_rank = self.master_rank)

    if tnt_distribute_dataset:
      distributed_dataset = ds.DistributedDataset(dataset = x,
                                                  num_ranks = self.comm_size,
                                                  rank = self.rank,
                                                  shuffle_seed = self.default_shuffle_seed)
      x = distributed_dataset.distribute_dataset_across_ranks()
    else:
      logging.getLogger().info("[rank %d] Automatic dataset distribution is disabled. \
Make sure the dataset is sharded manually across ranks." % (self.rank))
    return self.model.fit(x, **kwargs)
    
  def evaluate(self, x = None, **kwargs):
    test_dataset = ds.DistributedDataset(dataset = x,
                                         num_ranks = self.comm_size,
                                         rank = self.rank,
                                         shuffle_seed = self.default_shuffle_seed)
    x = test_dataset.distribute_dataset_across_ranks(is_training = False)
    return self.model.evaluate(x, **kwargs)

  def predict(self, x = None, **kwargs):
    test_dataset = ds.DistributedDataset(dataset = x,
                                         num_ranks = self.comm_size,
                                         rank = self.rank,
                                         shuffle_seed = self.default_shuffle_seed)
    x = test_dataset.distribute_dataset_across_ranks(is_training = False)
    return self.model.predict(x, **kwargs)
