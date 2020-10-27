import tensorflow as tf

class OptimizerWrapper(tf.keras.optimizers.Optimizer):
  def __init__(self, optimizer, name = None):
    self.optimizer = optimizer

    # overwrite the name of the inner optimizer
    if not name is None:
      self._name = name

  def __getattr__(self, name):
    return getattr(self.__dict__['optimizer'], name)
  
  def __setattr__(self, name, value):
    if name in ('optimizer'):
      self.__dict__[name] = value
    else:
      setattr(self.__dict__['optimizer'], name, value)
  
  def __delattr__(self, name):
    delattr(self.__dict__['optimizer'], name)

  # implement the missing methods by forwarding them to the inner optimizer implementations
  def _resource_apply_dense(self, *args, **kwargs):
    return self.optimizer._resource_apply_dense(*args, **kwargs)

  def _resource_apply_sparse(self, *args, **kwargs):
    raise NotImplementedError("[OptimizerWrapper] _resource_apply_sparse: Sparse tensors not supported.")

  def _create_slots(self, *args, **kwargs):
    return self.optimizer._create_slots(*args, **kwargs)

  def get_config(self):
    return self.optimizer.get_config()

