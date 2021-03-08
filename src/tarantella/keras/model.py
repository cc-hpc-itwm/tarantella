import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.keras.engine import training_utils
import tensorflow.keras.callbacks as tf_callbacks

import tarantella
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
import tarantella.datasets.distributed_dataset as ds
import tarantella.keras.callbacks as tnt_callbacks
from tarantella import logger


class Model(tf.keras.models.Model):
  def __init__(self, model):
    super().__init__()
    self.rank = tarantella.get_rank()
    self.comm_size = tarantella.get_size()

    self.model = model
    self.input_shapes = None
    self.done_broadcast = False
    self.compiled = False
    self.broadcaster = None
    self.barrier = tarantella.Barrier()

    self.orig_optimizer = None
    self.dist_optimizer = None
    self.default_shuffle_seed = 42

    # support for TF 2.0 -- 2.3
    self.tf_default_verbose = {'fit' : 1,
                               'evaluate' : 1,
                               'predict' : 0,
                              }

  ##############
  # Attributes #
  ##############
  @property
  def distribute_strategy(self):
    return None
  
  @property
  def dynamic(self):
    return self.model.dynamic
  
  @property
  def input_spec(self):
    return self.model.input_spec

  @property
  def layers(self):
    if hasattr(self, 'model'):
      return self.model.layers
    # condition needed for super(Model, self).__init__() to pass without error, 
    # as self.model does not exist at the time of init call
    else:
      return super().layers

  @property
  def losses(self):
    return self.model.losses

  @property
  def metrics(self):
    return self.model.metrics
  
  @property
  def metrics_names(self):
    return self.model.metrics_names
  
  @property
  def non_trainable_weights(self):
    return self.model.non_trainable_weights
  
  @property
  def output(self):
    return self.model.output

  @output.setter
  def output(self, value):
    self.model.output = value

  @property
  def run_eagerly(self):
    return self.model.run_eagerly
  
  @property
  def state_updates(self):
    return self.model.state_updates
  
  @property
  def stateful(self):
    return self.model.stateful
  
  @property
  def trainable_weights(self):
    return self.model.trainable_weights
  
  @property
  def weights(self):
    return self.model.weights

  #############
  # Functions #
  #############
  def add_loss(self, losses, *args, **kwargs):
    self.model.add_loss(losses, *args, **kwargs)
  
  def add_metric(self, value, *args, **kwargs):
    self.model.add_metric(value, *args, **kwargs)
  
  def build(self, input_shape):
    return self.model.build(input_shape)

  def call(self, inputs):
    return self.model.call(inputs)

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
    self.orig_optimizer_serialized = tf.keras.optimizers.serialize(optimizer)

    if 'experimental_run_tf_function' in kwargs:
      logger.info("Setting `experimental_run_tf_function` to False.")

    self.dist_optimizer = tarantella.distributed_optimizers.SynchDistributedOptimizer(self.orig_optimizer)
    return self.model.compile(optimizer = self.dist_optimizer,
                              loss = loss,
                              metrics = metrics,
                              loss_weights = loss_weights,
                              sample_weight_mode = sample_weight_mode,
                              weighted_metrics = weighted_metrics,
                              experimental_run_tf_function = False,
                              **kwargs)

  def compute_mask(self, inputs, mask):
    return self.model.compute_mask(inputs, mask)
  
  def compute_output_shape(self, input_shape):
    return self.model.compute_output_shape(input_shape)

  def evaluate(self,
               x = None,
               y = None,
               callbacks = None,
               tnt_micro_batch_size = None,
               tnt_distribute_dataset = True,
               **kwargs):
    self._setup_for_execution('evaluate', x, y, kwargs)
    processed_callbacks = self._preprocess_callbacks(callbacks)

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

    return self.model.evaluate(x, callbacks = processed_callbacks, **kwargs)

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
    self._setup_for_execution('fit', x, y, kwargs)
    processed_callbacks = self._preprocess_callbacks(callbacks)

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
                          callbacks = processed_callbacks,
                          **kwargs)

  @classmethod
  def from_config(cls, config):
    try:
      keras_model = tf.keras.Model.from_config(config)
      logger.info("Loaded model from `keras.Model`.")
    except:
      try:
        keras_model = tf.keras.Sequential.from_config(config)
        logger.info("Loaded model from `keras.Sequential`.")
      except:
        raise RuntimeError("""[tnt.Model.from_config] Cannot load
              model; provided configuration is neither a `keras.Model`
              nor a `keras.Sequential` model.""")
    return cls(keras_model)

  def get_config(self):
    return self.model.get_config()

  def get_layer(self, name=None, index=None):
    return self.model.get_layer(name, index)
  
  def get_weights(self):
    if not self.model.built:
      if not self.input_shapes:
        raise RuntimeError("""Cannot get weights before initializition.
        Please call "tnt.Model.build()" or "tnt.Model.fit()" first.
        """)
      self.model.build(self.input_shapes)
    return self.model.get_weights()

  def load_weights(self, filepath, **kwargs):
    # loaded weights from the same source will be identical on all ranks
    self.done_broadcast = True
    return self.model.load_weights(filepath = filepath, **kwargs)
  
  def predict(self,
              x = None,
              callbacks = None,
              tnt_micro_batch_size = None,
              tnt_distribute_dataset = True,
              **kwargs):
    self._setup_for_execution('predict', x, None, kwargs)
    processed_callbacks = self._preprocess_callbacks(callbacks)

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
    return self.model.predict(x, callbacks = processed_callbacks, **kwargs)

  def reset_metrics(self):
    self.model.reset_metrics()

  def reset_states(self):
    self.model.reset_states()
  
  def save(self, filepath, tnt_save_all_devices = False, **kwargs):
    if tnt_save_all_devices:
      self._save(filepath, kwargs)
    else:
      if tarantella.is_master_rank():
        self._save(filepath, kwargs)
    # make sure, every rank can load the model after function exit
    self.barrier.synchronize()

  def save_weights(self, filepath, tnt_save_all_devices = False, **kwargs):
    if tnt_save_all_devices:
      self.model.save_weights(filepath, **kwargs)
    else:
      if tarantella.is_master_rank():
        self.model.save_weights(filepath, **kwargs)
    # make sure, every rank can load the model after function exit
    self.barrier.synchronize()

  def set_weights(self, weights):
    self.model.set_weights(weights)
    self._broadcast_weights()
    self.done_broadcast = True
    
  def summary(self, *args, **kwargs):
    if tarantella.global_tnt_config.output_on_all_devices:
      self.model.summary(*args, **kwargs)
    else:
      if tarantella.is_master_rank():
        self.model.summary(*args, **kwargs)

  def to_json(self, **kwargs):
    return self.model.to_json(**kwargs)

  def to_yaml(self, **kwargs):
    return self.model.to_yaml(**kwargs)
  
  ####################
  # Helper functions #
  ####################
  def _save(self, filepath, args_dict):
    if self.compiled == False:
      #clean the optimizer if not compiled.
      self.model.optimizer = None
      self.model.save(filepath = filepath, **args_dict)
    else:
      # 1. Set the optimizer of the underlying `keras.Model` using the original optimizer
      deserialized_optimizer = tf.keras.optimizers.deserialize(self.orig_optimizer_serialized)
      self._set_internal_optimizer(deserialized_optimizer)
      # 2. Save the model as `keras.Model` with standard Keras optimizer
      self.model.save(filepath = filepath, **args_dict)
      # 3. Reset the underlying optimizer to the distributed optimzier
      self._set_internal_optimizer(self.dist_optimizer)

  def _set_internal_optimizer(self, optimizer):
    if hasattr(self, '_get_optimizer'):
      self.model.optimizer = self._get_optimizer(optimizer)
    elif hasattr(self, '_set_optimizer'):
      #for Sequential model with TF 2.0/2.1
      self.model._set_optimizer(optimizer)
    else:
      raise RuntimeError(
      "[tnt.Model._set_internal_optimizer] Cannot set optimizer for the internal `keras.Model`")

  def _setup_for_execution(self, exec_type, x, y, args_dict):
    self._assert_compile_has_been_called()
    self._set_verbose_all_ranks(exec_type, args_dict)
    self._validate_datasets(x, y)
    self._validate_batch_size_argument(exec_type, args_dict)
    self._set_input_shapes(x)
    self._broadcast_weights_if_necessary()

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

  def _preprocess_callbacks(self, callbacks):
    callbacks = callbacks or []

    self._add_default_callbacks(callbacks)
    self._to_tnt_callbacks(callbacks)

    return callbacks

  def _add_default_callbacks(self, callbacks):
    history_obj_exists = False
    for callback in callbacks:
      if isinstance(callback, tf_callbacks.History):
        history_obj_exists = True

    if history_obj_exists is False:
      callbacks.append(tf_callbacks.History())

  def _to_tnt_callbacks(self, callbacks):
    remove_tensorboard_index = None

    for index, callback in enumerate(callbacks):
      if isinstance(callback, tf_callbacks.ModelCheckpoint):
        tnt_callback = tnt_callbacks.ModelCheckpoint(keras_callback = callback,
                                                     underlying_optimizer = self.orig_optimizer,
                                                     distributed_optimizer = self.dist_optimizer)
        callbacks[index] = tnt_callback

      elif isinstance(callback, tf_callbacks.LearningRateScheduler):
        tnt_callback = tnt_callbacks.LearningRateScheduler(keras_callback = callback)
        callbacks[index] = tnt_callback

      elif isinstance(callback, tf_callbacks.TensorBoard):
        if tarantella.global_tnt_config.tensorboard_on_all_devices:
          callback.log_dir += '/rank_{}'.format(self.rank)
        else:
          if not tarantella.is_master_rank():
            remove_tensorboard_index = index

      elif isinstance(callback, tf_callbacks.History):
        hist_callback = tnt_callbacks.History(keras_callback = callback)
        callbacks[index] = hist_callback
      
      elif isinstance(callback, tf_callbacks.EarlyStopping):
        early_stopping_callback = tnt_callbacks.EarlyStopping(keras_callback = callback)
        callbacks[index] = early_stopping_callback
      
    if remove_tensorboard_index is not None:
      del callbacks[remove_tensorboard_index]

def connect_ancillary_layers(model, created_layers):
  raise AttributeError('Not supported by tarantella model. '
                       'Call `connect_ancillary_layers` on keras '
                       ' model before calling `tnt.Model()` instead.')
