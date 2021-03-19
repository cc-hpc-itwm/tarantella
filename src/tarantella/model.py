import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.keras.engine import training_utils

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
import tarantella.datasets.distributed_dataset as ds
import tarantella.callbacks as tnt_callbacks
from tarantella import logger

model_implemented_methods = ['model', 'rank', 'comm_size',
                             'call', 'build', 'done_broadcast', 'set_weights', 'load_weights',
                             'get_weights', '_broadcast_weights_if_necessary', '_broadcast_weights',
                             'broadcaster', 'default_shuffle_seed',
                             'orig_optimizer', 'orig_loss', 'orig_metrics',
                             'orig_loss_weights', 'orig_sample_weight_mode', 'orig_weighted_metrics']

class Model(tf.keras.models.Model):
  def __init__(self, model):
    self.rank = tarantella.get_rank()
    self.comm_size = tarantella.get_size()

    self.model = model
    self.input_shapes = None
    self.done_broadcast = False
    self.compiled = False
    self.broadcaster = None
    self.barrier = tarantella.Barrier()

    self.orig_optimizer = None
    self.orig_loss = None
    self.orig_metrics = None
    self.orig_loss_weights = None
    self.orig_sample_weight_mode = None
    self.orig_weighted_metrics = None

    self.dist_optimizer = None
    self.default_shuffle_seed = 42

    # support for TF 2.0 -- 2.3
    self.tf_default_verbose = {'fit' : 1,
                               'evaluate' : 1,
                               'predict' : 0,
                              }

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
    self.compiled = True

    # Store original parameters to save the model later
    self.orig_optimizer = optimizer
    self.orig_loss = loss
    self.orig_metrics = metrics
    self.orig_loss_weights = loss_weights
    self.orig_sample_weight_mode = sample_weight_mode
    self.orig_weighted_metrics = weighted_metrics

    self.dist_optimizer = tarantella.distributed_optimizers.SynchDistributedOptimizer(self.orig_optimizer)
    return self.model.compile(optimizer = self.dist_optimizer,
                              loss = self.orig_loss,
                              metrics = self.orig_metrics,
                              loss_weights = self.orig_loss_weights,
                              sample_weight_mode = self.orig_sample_weight_mode,
                              weighted_metrics = self.orig_weighted_metrics,
                              **kwargs)

  def fit(self,
          x = None,
          y = None,
          callbacks = None,
          validation_data = None,
          tnt_micro_batch_size = None,
          tnt_validation_micro_batch_size = None,
          tnt_distribute_dataset = True,
          tnt_distribute_validation_dataset = True,
          **kwargs):
    self._setup_for_execution('fit', x, y, callbacks, kwargs)

    if tnt_distribute_dataset:
      distributed_x = ds.DistributedDataset(dataset = x,
                                            num_ranks = self.comm_size,
                                            rank = self.rank,
                                            shuffle_seed = self.default_shuffle_seed)
      x = distributed_x.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = True)
    else:
      logger.info("Automatic dataset distribution is disabled."
                  "Make sure the dataset is sharded manually across ranks.")

    # Always switch off shuffling
    kwargs["shuffle"] = False

    if validation_data:
      if tnt_distribute_validation_dataset:
        distributed_validation_data = ds.DistributedDataset(dataset = validation_data,
                                                            num_ranks = self.comm_size,
                                                            rank = self.rank,
                                                            shuffle_seed = self.default_shuffle_seed)
        validation_data = distributed_validation_data.distribute_dataset_across_ranks(
              user_micro_batch_size = tnt_validation_micro_batch_size,
              is_training = False)
      else:
        logger.info("Automatic distribution for the validation dataset is disabled.")

    return self.model.fit(x,
                          validation_data = validation_data,
                          callbacks = callbacks,
                          **kwargs)
    
  def evaluate(self,
               x = None,
               y = None,
               callbacks = None,
               tnt_micro_batch_size = None,
               tnt_distribute_dataset = True,
               **kwargs):
    self._setup_for_execution('evaluate', x, y, callbacks, kwargs)

    if tnt_distribute_dataset:
      test_dataset = ds.DistributedDataset(dataset = x,
                                          num_ranks = self.comm_size,
                                          rank = self.rank,
                                          shuffle_seed = self.default_shuffle_seed)
      x = test_dataset.distribute_dataset_across_ranks(
              user_micro_batch_size = tnt_micro_batch_size,
              is_training = False)
    else:
      logger.info("Automatic dataset distribution is disabled.")

    return self.model.evaluate(x, callbacks = callbacks, **kwargs)

  def predict(self,
              x = None,
              callbacks = None,
              tnt_micro_batch_size = None,
              tnt_distribute_dataset = True,
              **kwargs):
    self._setup_for_execution('predict', x, None, callbacks, kwargs)

    if tnt_distribute_dataset:
      test_dataset = ds.DistributedDataset(dataset = x,
                                           num_ranks = self.comm_size,
                                           rank = self.rank,
                                           shuffle_seed = self.default_shuffle_seed)
      x = test_dataset.distribute_dataset_across_ranks(
               user_micro_batch_size = tnt_micro_batch_size,
               is_training = False)
    else:
      logger.info("Automatic dataset distribution is disabled.")
    return self.model.predict(x, callbacks = callbacks, **kwargs)

  def get_config(self):
    return self.model.get_config()

  @classmethod
  def from_config(cls, config):
    keras_model = tf.keras.Model.from_config(config)
    return cls(keras_model)

  def to_json(self, **kwargs):
    return self.model.to_json(**kwargs)

  def to_yaml(self, **kwargs):
    return self.model.to_yaml(**kwargs)

  def save_weights(self, filepath, tnt_save_all_devices = False, **kwargs):
    if tnt_save_all_devices:
      self.model.save_weights(filepath, **kwargs)
    else:
      if tarantella.is_master_rank():
        self.model.save_weights(filepath, **kwargs)
    # make sure, every rank can load the model after function exit
    self.barrier.synchronize()

  def load_weights(self, filepath, **kwargs):
    # loaded weights from the same source will be identical on all ranks
    self.done_broadcast = True
    return self.model.load_weights(filepath = filepath, **kwargs)
  
  def set_weights(self, weights):
    self.model.set_weights(weights)
    self._broadcast_weights()
    self.done_broadcast = True
    
  def get_weights(self):
    if not self.model.built:
      if not self.input_shapes:
        raise RuntimeError("""Cannot get weights before initializition.
        Please call "tnt.Model.build()" or "tnt.Model.fit()" first.
        """)
      self.model.build(self.input_shapes)
    return self.model.get_weights()

  def save(self, filepath, tnt_save_all_devices = False, **kwargs):
    if tnt_save_all_devices:
      self._save(filepath, kwargs)
    else:
      if tarantella.is_master_rank():
        self._save(filepath, kwargs)
    # make sure, every rank can load the model after function exit
    self.barrier.synchronize()

  def _save(self, filepath, args_dict):
    # 1. Re-compile underlying `Keras.model` w/ underlying optimizer
    self.model.compile(optimizer = self.orig_optimizer,
                       loss = self.orig_loss,
                       metrics = self.orig_metrics,
                       loss_weights = self.orig_loss_weights,
                       sample_weight_mode = self.orig_sample_weight_mode,
                       weighted_metrics = self.orig_weighted_metrics)

    # 2. Save the model as `Keras.Model` with standard Keras optimizer
    self.model.save(filepath = filepath, **args_dict)

    # 3. Re-compile the Tarantella Model
    self.compile(optimizer = self.orig_optimizer,
                 loss = self.orig_loss,
                 metrics = self.orig_metrics,
                 loss_weights = self.orig_loss_weights,
                 sample_weight_mode = self.orig_sample_weight_mode,
                 weighted_metrics = self.orig_weighted_metrics)

  def summary(self, *args, **kwargs):
    if tarantella.global_tnt_config.output_on_all_devices:
      self.model.summary(*args, **kwargs)
    else:
      if tarantella.is_master_rank():
        self.model.summary(*args, **kwargs)

  def _setup_for_execution(self, exec_type, x, y, callbacks, args_dict):
    self._assert_compile_has_been_called()
    self._set_verbose_all_ranks(exec_type, args_dict)
    self._validate_datasets(x, y)
    self._validate_batch_size_argument(exec_type, args_dict)
    self._set_input_shapes(x)
    self._broadcast_weights_if_necessary()
    self._add_default_callbacks(callbacks)
    self._preprocess_callbacks(callbacks)

  def _assert_compile_has_been_called(self):
    if self.compiled == False:
      raise RuntimeError("`tnt.Model` has to be compiled first "
                         "using `tnt.Model.compile`")

  def _set_verbose_all_ranks(self, exec_type, args_dict):
    if not 'verbose' in args_dict:
      args_dict['verbose'] = self.tf_default_verbose[exec_type]
    if not tarantella.global_tnt_config.output_on_all_devices:
      if not tarantella.is_master_rank():
        args_dict['verbose'] = 0

  def _validate_datasets(self, x, y):
    if not isinstance(x, tf.data.Dataset) or not y is None:
      raise RuntimeError("tnt.Model only supports `tf.data.Dataset`",
                         "for `x` and `None` for y.")

  def _validate_batch_size_argument(self, exec_type, args_dict):
    if 'batch_size' in args_dict:
      raise KeyError("tnt.Model does not support `batch_size` argument in %s" % exec_type)

    if 'validation_batch_size' in args_dict and exec_type == 'fit':
      raise KeyError("tnt.Model.fit does not support `validation_batch_size` argument")

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
    root_rank = tarantella.get_master_rank()
    if not self.broadcaster:
      weights = self.get_weights()
      self.broadcaster = tarantella.TensorBroadcaster(weights, root_rank)

    if self.rank == root_rank:
      weights = self.get_weights()
      self.broadcaster.broadcast(weights)
    else:
      new_weights = self.broadcaster.broadcast()
      self.model.set_weights(new_weights)

    self.done_broadcast = True

  def _add_default_callbacks(self, callbacks):
    if callbacks is not None:
      callbacks.append(tf.keras.callbacks.History())
    else:
      callbacks = tf.keras.callbacks.History()

  def _preprocess_callbacks(self, callbacks):
    if callbacks is not None:
      remove_tensorboard_index = None

      for index, callback in enumerate(callbacks):
        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
          tnt_callback = tnt_callbacks.TntModelCheckpoint(keras_model_checkpoint = callback,
                                                          underlying_optimizer = self.orig_optimizer,
                                                          distributed_optimizer = self.dist_optimizer)
          callbacks[index] = tnt_callback

        elif isinstance(callback, tf.keras.callbacks.LearningRateScheduler):
          if not tarantella.global_tnt_config.output_on_all_devices:
            if not tarantella.is_master_rank():
              callback.verbose = 0

        elif isinstance(callback, tf.keras.callbacks.TensorBoard):
          if tarantella.global_tnt_config.tensorboard_on_all_devices:
            callback.log_dir += '/rank_{}'.format(self.rank)
          else:
            if not tarantella.is_master_rank():
              remove_tensorboard_index = index

        elif isinstance(callback, tf.keras.callbacks.CSVLogger):
          csv_callback = tnt_callbacks.TntCSVLogger(keras_csvlogger = callback)
          callbacks[index] = csv_callback

        elif isinstance(callback, tf.keras.callbacks.History):
          hist_callback = tnt_callbacks.TntHistory(keras_history = callback)
          callbacks[index] = hist_callback

      if remove_tensorboard_index is not None:
        del callbacks[remove_tensorboard_index]
