import logging
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.keras.engine import training_utils

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
import tarantella.datasets.distributed_dataset as ds

model_implemented_methods = ['model', 'rank', 'comm_size', '_master_rank', 'threshold',
                             'call', 'build', 'done_broadcast', 'set_weights', 'load_weights',
                             'broadcast_weights_if_necessary', 'broadcast_weights',
                             'broadcaster', 'default_shuffle_seed']

class TarantellaModel(tf.keras.models.Model):
  def __init__(self, model, _fusion_threshold_bytes = 32768):
    if not tarantella.global_context:
      raise RuntimeError("""Cannot initialize a TarantellaModel before the Tarantella library.
      Please call "tarantella.init() first."
      """)
    self._master_rank = 0
    self.rank = tarantella.get_rank()
    self.comm_size = tarantella.get_size()

    self.model = model
    self.done_broadcast = False
    self.broadcaster = None

    self.threshold = _fusion_threshold_bytes
    self.default_shuffle_seed = 42

  @property
  def master_rank(self):
    return self._master_rank

  def call(self, inputs):
    return self.model.call(inputs)

  def build(self, input_shape):
    return self.model.build(input_shape)

  def __getattr__(self, name):
    if name in model_implemented_methods or \
       'model' not in self.__dict__:
      return getattr(self.__dict__, name)
    return getattr(self.__dict__['model'], name)
  
  def __setattr__(self, name, value):
    if name in model_implemented_methods or \
       'model' not in self.__dict__:
      self.__dict__[name] = value
    else:
      setattr(self.__dict__['model'], name, value)
  
  def __delattr__(self, name):
    if name in model_implemented_methods or \
       'model' not in self.__dict__:
      delattr(self.__dict__, name)
    delattr(self.__dict__['model'], name)

  def get_weights(self):
    return self.model.get_weights()

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    self.done_broadcast = False
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
          x = None,
          tnt_micro_batch_size = None,
          tnt_distribute_dataset = True,
          **kwargs):
    self.broadcast_weights_if_necessary()

    if tnt_distribute_dataset:
      distributed_dataset = ds.DistributedDataset(dataset = x,
                                                  num_ranks = self.comm_size,
                                                  rank = self.rank,
                                                  shuffle_seed = self.default_shuffle_seed)
      x = distributed_dataset.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = True)
    else:
      logging.getLogger().info("[rank %d] Automatic dataset distribution is disabled. \
Make sure the dataset is sharded manually across ranks." % (self.rank))
    return self.model.fit(x, **kwargs)
    
  def evaluate(self,
               x = None,
               tnt_micro_batch_size = None,
               **kwargs):
    self.broadcast_weights_if_necessary()

    test_dataset = ds.DistributedDataset(dataset = x,
                                         num_ranks = self.comm_size,
                                         rank = self.rank,
                                         shuffle_seed = self.default_shuffle_seed)
    x = test_dataset.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = False)
    return self.model.evaluate(x, **kwargs)

  def predict(self,
              x = None,
              tnt_micro_batch_size = None,
              **kwargs):
    self.broadcast_weights_if_necessary()

    test_dataset = ds.DistributedDataset(dataset = x,
                                         num_ranks = self.comm_size,
                                         rank = self.rank,
                                         shuffle_seed = self.default_shuffle_seed)
    x = test_dataset.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = False)
    return self.model.predict(x, **kwargs)

  def load_weights(self, *args, **kwargs):
    # loaded weights from the same source will be identical on all ranks
    self.done_broadcast = True
    return self.model.load_weights(*args, **kwargs)
  
  def set_weights(self, *args, **kwargs):
    self.model.set_weights(*args, **kwargs)
    self.broadcast_weights()
    
  def broadcast_weights_if_necessary(self):
    if not self.done_broadcast:
      self.broadcast_weights()

  def broadcast_weights(self):
    # FIXME: weights may not have been available/initialized in TF yet -> add build(input_shape)
    weights = self.get_weights()

    if not self.broadcaster:
      self.broadcaster = tarantella.TensorBroadcaster(weights, self._master_rank)

    self.broadcaster.broadcast(weights)
    self.model.set_weights(weights)

    self.done_broadcast = True