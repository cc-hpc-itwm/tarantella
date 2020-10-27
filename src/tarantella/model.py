import tensorflow as tf

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers

class TarantellaModel(tf.keras.models.Model):
  def __init__(self, model, _fusion_threshold_bytes = 32768):
    if not tarantella.global_context:
      raise RuntimeError("""Cannot initialize a TarantellaModel before the Tarantella library.
      Please call "tarantella.init() first."
      """)
    self.model = model
    self.threshold = _fusion_threshold_bytes

  def __getattr__(self, name):
    return getattr(self.__dict__['model'], name)
  
  def __setattr__(self, name, value):
    if name in ('model'):
      self.__dict__[name] = value
    else:
      setattr(self.__dict__['model'], name, value)
  
  def __delattr__(self, name):
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
                              
  def fit(self, *args, **kwargs):
    # Broadcast initial weights to all processes
    tarantella.broadcast_model_weights(self.model, root_rank = 0)
    return self.model.fit(*args, **kwargs)
    
  def evaluate(self, *args, **kwargs):
    return self.model.evaluate(*args, **kwargs)

  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)
