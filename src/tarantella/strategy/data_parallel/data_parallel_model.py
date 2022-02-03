import tensorflow as tf

import tarantella as tnt
import tarantella.keras.utilities as utilities
import tarantella.strategy.parallel_model as parallel_model
import tarantella.utilities.tf_version as version_utils
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
from tarantella import logger

class DataParallelModel(parallel_model.ParallelModel):
  def __init__(self, model, group = tnt.Group()):
    super().__init__(model = model, group = group)
    self.input_shapes = None
    self.done_broadcast = False
    self.compiled = False
    self.broadcaster = None
    self.barrier = tnt.Barrier(group = self.group)

    self.dist_optimizer = None
    self.default_shuffle_seed = 42

  @property
  def metrics(self):
    return self.model.metrics

  @property
  def metrics_names(self):
    return self.model.metrics_names

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
    logger.info("[DataParallelModel] compile.")
    if isinstance(optimizer, dict):
      optimizer = tf.keras.optimizers.deserialize(optimizer)
    elif isinstance(optimizer, str):
      config = {'class_name': optimizer, 'config': {}}
      optimizer = tf.keras.optimizers.deserialize(config)
    self.dist_optimizer = tnt.distributed_optimizers.SynchDistributedOptimizer(optimizer, group = self.group)

    kwargs = self._preprocess_compile_kwargs(kwargs)
    return self.model.compile(optimizer = self.dist_optimizer,
                              loss = loss,
                              metrics = metrics,
                              loss_weights = loss_weights,
                              sample_weight_mode = sample_weight_mode,
                              weighted_metrics = weighted_metrics,
                              **kwargs)

  def evaluate(self,
               x = None,
               y = None,
               callbacks = None,
               tnt_micro_batch_size = None,
               tnt_distribute_dataset = True,
               **kwargs):
    self._setup_for_execution('evaluate', x, y, kwargs)
    processed_callbacks = utilities._preprocess_callbacks(callbacks, self.group,
                                                          exec_type = 'evaluate',
                                                          verbose = kwargs.get('verbose', None))

    if tnt_distribute_dataset:
      test_dataset = tnt.data.Dataset(dataset = x,
                                      num_ranks = self.group.size,
                                      rank = self.rank,
                                      shuffle_seed = self.default_shuffle_seed)
      x = test_dataset.distribute_dataset_across_ranks(
              user_micro_batch_size = tnt_micro_batch_size,
              is_training = False)
      self._validate_micro_batch_size_for_batch_normalization(test_dataset.micro_batch_size)
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
    processed_callbacks = utilities._preprocess_callbacks(callbacks, self.group,
                                                          exec_type = 'fit',
                                                          verbose = kwargs.get('verbose', None))

    if tnt_distribute_dataset:
      # Distribute dataset into micro-batches among ranks by taking into account
      # all possible cases of splitting the dataset:
      #
      # 1. Batch size
      # a. `batch_size` is a multiple of the number of ranks
      #     => identical `micro_batch_size` for all ranks
      # b. `batch_size` is not a multiple of the number of ranks
      #     => different ranks have different `micro_batch_size`s and
      #        locally computed gradients need to be scaled by a factor to
      #        account for the differences
      # c. `batch_size` < number of ranks
      #     => raise Error
      #
      # 2. Last batch within epoch
      # a. the last batch in the dataset is incomplete, but dataset is batched
      #    with `drop_remainder = True`
      #     => the last batch is dropped
      # b. the last batch in the dataset is incomplete with `drop_remainder = False`
      #     - number of samples in the last batch is smaller than `num_ranks`,
      #         => pad the dataset with a number of zeroed samples to ensure that each rank
      #            has one sample, so that they all see the same number of iterations in an epoch;
      #            the fake samples will be filtered out from the final gradient computation by
      #            assigning them `micro_batch_size = 0`
      #     - number of samples in the last batch is >= `num_ranks`
      #         => last batch can be considered a new `batch_size`, which will be handled as above (in 1.),
      #            both for computing the `micro_batch_size` and the `scaling_factor`
      distributed_x = tnt.data.Dataset(dataset = x,
                                       num_ranks = self.group.size,
                                       rank = self.group.to_group_rank(self.rank),
                                       shuffle_seed = self.default_shuffle_seed)
      x = distributed_x.distribute_dataset_across_ranks(
            user_micro_batch_size = tnt_micro_batch_size,
            is_training = True)
      self._validate_micro_batch_size_for_batch_normalization(distributed_x.micro_batch_size)

      # if different ranks have different micro-batch sizes, the gradients need rescaling
      dataset_callback = distributed_x.get_gradient_scaling_callback()
      if dataset_callback:
        processed_callbacks.append(dataset_callback)

    else:
      logger.info("Automatic dataset distribution is disabled."
                  "Make sure the dataset is sharded manually across ranks.")

    # Always switch off shuffling
    kwargs["shuffle"] = False

    if validation_data:
      if tnt_distribute_validation_dataset:
        distributed_validation_data = tnt.data.Dataset(dataset = validation_data,
                                                       num_ranks = self.group.size,
                                                       rank = self.rank,
                                                       shuffle_seed = self.default_shuffle_seed)
        validation_data = distributed_validation_data.distribute_dataset_across_ranks(
              user_micro_batch_size = tnt_validation_micro_batch_size,
              is_training = False)
        self._validate_micro_batch_size_for_batch_normalization(distributed_validation_data.micro_batch_size)
      else:
        logger.info("Automatic distribution for the validation dataset is disabled.")

    return self.model.fit(x = x,
                          validation_data = validation_data,
                          callbacks = processed_callbacks,
                          **kwargs)

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
    result = self.model.load_weights(filepath = filepath, **kwargs)
    self.barrier.execute()
    return result

  def predict(self,
              x = None,
              callbacks = None,
              tnt_micro_batch_size = None,
              tnt_distribute_dataset = True,
              **kwargs):
    self._setup_for_execution('predict', x, None, kwargs)
    processed_callbacks = utilities._preprocess_callbacks(callbacks, self.group,
                                                          exec_type = 'predict',
                                                          verbose = kwargs.get('verbose', None))

    if tnt_distribute_dataset:
      test_dataset = tnt.data.Dataset(dataset = x,
                                      num_ranks = self.group.size,
                                      rank = self.rank,
                                      shuffle_seed = self.default_shuffle_seed)
      x = test_dataset.distribute_dataset_across_ranks(
               user_micro_batch_size = tnt_micro_batch_size,
               is_training = False)
      self._validate_micro_batch_size_for_batch_normalization(test_dataset.micro_batch_size)
    else:
      logger.info("Automatic dataset distribution is disabled.")
    return self.model.predict(x, callbacks = processed_callbacks, **kwargs)

  def save(self, filepath, tnt_save_all_devices = False, **kwargs):
    self._save_to_file(tnt_save_all_devices, save_function = self._save_tnt_model,
                       filepath = filepath, **kwargs)

  def save_weights(self, filepath, tnt_save_all_devices = False, **kwargs):
    self._save_to_file(tnt_save_all_devices, save_function = self.model.save_weights,
                       filepath = filepath, **kwargs)

  def set_weights(self, weights):
    self.model.set_weights(weights)
    self._broadcast_weights()
    self.done_broadcast = True

  ####################
  # Helper functions #
  ####################
  def _save_to_file(self, tnt_save_all_devices, save_function,
                    filepath, **kwargs):
    if tnt_save_all_devices:
      save_function(filepath, kwargs)
    else:
      if tnt.is_master_rank():
        save_function(filepath, kwargs)
    # make sure that every rank can load the model after function exit
    self.barrier.execute()

  def _save_tnt_model(self, filepath, args_dict):
    if self.compiled == False:
      self.model.save(filepath = filepath, **args_dict)
    else:
      self._set_internal_optimizer(self.dist_optimizer.underlying_optimizer)
      self.model.save(filepath = filepath, **args_dict)
      self._set_internal_optimizer(self.dist_optimizer)

  def _set_internal_optimizer(self, optimizer):
    utilities._set_model_optimizer(self.model, optimizer)

  def _setup_for_execution(self, exec_type, x, y, args_dict):
    self._assert_compile_has_been_called()
    self._validate_datasets(x, y)
    self._validate_batch_size_argument(exec_type, args_dict)
    self._set_input_shapes(x)
    self._broadcast_weights_if_necessary()

  def _assert_compile_has_been_called(self):
    if self.compiled == False:
      raise RuntimeError("`tnt.Model` has to be compiled first "
                         "using `tnt.Model.compile`")

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
    root_rank = 0 # rank 0 within the group
    if not self.broadcaster:
      self.broadcaster = tnt.TensorBroadcaster(inputs = self.get_weights(),
                                               group = self.group, root_rank = root_rank)

    if self.rank == self.group.to_global_rank(root_rank):
      self.broadcaster.broadcast(self.get_weights())
    else:
      new_weights = self.broadcaster.broadcast()
      self.model.set_weights(new_weights)

    self.done_broadcast = True

  def _preprocess_compile_kwargs(self, kwargs):
    if version_utils.tf_version_below_equal('2.1'):
      kwargs['experimental_run_tf_function'] = False
      logger.info("Set `experimental_run_tf_function` to False.")
    return kwargs

  def _validate_micro_batch_size_for_batch_normalization(self, micro_batch_size):
    if micro_batch_size < 16:
      for layer in self.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
          logger.warn("Micro batch size should be at least 16 when using Batch Normalization.")
          return

  def close(self):
    del self.broadcaster
    del self.barrier
    del self.dist_optimizer


