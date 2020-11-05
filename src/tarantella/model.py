import logging
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.keras.engine import training_utils

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
import tarantella.datasets.distributed_dataset as ds

model_implemented_methods = ['model', 'rank', 'comm_size', '_master_rank',
                             'call', 'build', 'done_broadcast', 'set_weights', 'load_weights',
                             'get_weights', '_broadcast_weights_if_necessary', '_broadcast_weights',
                             'broadcaster', 'default_shuffle_seed']

class TarantellaModel(tf.keras.models.Model):
  def __init__(self, model):
    if not tarantella.global_context:
      raise RuntimeError("""Cannot initialize a TarantellaModel before the Tarantella library.
      Please call "tarantella.init()" first.
      """)
    self._master_rank = 0
    self.rank = tarantella.get_rank()
    self.comm_size = tarantella.get_size()

    self.model = model
    self.input_shapes = None
    self.done_broadcast = False
    self.broadcaster = None

    self.default_shuffle_seed = 42

    # support for TF 2.0 -- 2.3
    self.tf_default_verbose = {'fit' : 1,
                               'evaluate' : 1,
                               'predict' : 0,
                              }

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

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    self.done_broadcast = False
    optimizer = tarantella.distributed_optimizers.SynchDistributedOptimizer(optimizer)
    return self.model.compile(optimizer = optimizer,
                              loss = loss,
                              metrics = metrics,
                              loss_weights = loss_weights,
                              sample_weight_mode = sample_weight_mode,
                              weighted_metrics = weighted_metrics,
                              **kwargs)

  def fit(self,
          x = None,
          y = None,
          validation_data = None,
          tnt_micro_batch_size = None,
          tnt_validation_micro_batch_size = None,
          tnt_distribute_dataset = True,
          **kwargs):
    self._setup_for_execution('fit', x, y, kwargs)

    if tnt_distribute_dataset:
      distributed_x = ds.DistributedDataset(dataset = x,
                                            num_ranks = self.comm_size,
                                            rank = self.rank,
                                            shuffle_seed = self.default_shuffle_seed)
      x = distributed_x.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = True)
    else:
      logging.getLogger().info(
        "[rank %d] Automatic dataset distribution is disabled." % (self.rank),
        "Make sure the dataset is sharded manually across ranks.")

    # Always switch off shuffling
    kwargs["shuffle"] = False

    if validation_data:
      distributed_validation_data = ds.DistributedDataset(dataset = validation_data,
                                                          num_ranks = self.comm_size,
                                                          rank = self.rank,
                                                          shuffle_seed = self.default_shuffle_seed)
      validation_data = distributed_validation_data.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_validation_micro_batch_size,
            is_training = False)

    return self.model.fit(x, validation_data = validation_data, **kwargs)
    
  def evaluate(self,
               x = None,
               y = None,
               tnt_micro_batch_size = None,
               **kwargs):
    self._setup_for_execution('evaluate', x, y, kwargs)

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
    self._setup_for_execution('predict', x, None, kwargs)

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
    self._broadcast_weights()
    self.done_broadcast = True
    
  def get_weights(self, *args, **kwargs):
    if not self.model.built:
      if not self.input_shapes:
        raise RuntimeError("""Cannot get weights before initializition.
        Please call "tnt.Model.build()" or "tnt.model.TarantellaModel.fit()" first.
        """)
      self.model.build(self.input_shapes)
    return self.model.get_weights(*args, **kwargs)

  def _setup_for_execution(self, exec_type, x, y, args_dict):
    self._set_verbose_all_ranks(exec_type, args_dict)
    self._validate_datasets(x, y)
    self._set_input_shapes(x)
    self._broadcast_weights_if_necessary()

  def _set_verbose_all_ranks(self, exec_type, args_dict):
    if not 'verbose' in args_dict:
      args_dict['verbose'] = self.tf_default_verbose[exec_type]
    if not tarantella.global_tnt_config.log_on_all_devices:
      if self.rank != self._master_rank:
        args_dict['verbose'] = 0

  def _validate_datasets(self, x, y):
    if not isinstance(x, tf.data.Dataset) or not y is None:
      raise RuntimeError("tnt.model.TarantellaModel only supports `tf.data.Dataset`\
 for `x` and `None` for y.")

  def _set_input_shapes(self, dataset):
    if isinstance(dataset.element_spec, tf.TensorSpec):
      self.input_shapes = dataset.element_spec.shape
    elif isinstance(dataset.element_spec[0], tf.TensorSpec): # (input, outputs)
      self.input_shapes = dataset.element_spec[0].shape
    else: # ((input0, ..., input_n), outputs)
      self.input_shapes = [elem_spec.shape for elem_spec in dataset.element_spec[0]]

  def _broadcast_weights_if_necessary(self):
    if not self.done_broadcast:
      self._broadcast_weights()

  def _broadcast_weights(self):
    weights = self.get_weights()

    if not self.broadcaster:
      self.broadcaster = tarantella.TensorBroadcaster(weights, self._master_rank)

    self.broadcaster.broadcast(weights)
    self.model.set_weights(weights)

    self.done_broadcast = True
