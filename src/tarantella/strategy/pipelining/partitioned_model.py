import tarantella.strategy.parallel_model as parallel_model
import tarantella.strategy.pipelining.partition_generator as pgen
import tarantella.strategy.pipelining.rank_mapper as rmapper
import tarantella.strategy.pipelining.core_model_builder as cm_builder
import tarantella.strategy.pipelining.shared_model_builder as sm_builder
import tarantella.strategy.pipelining.microbatched_model_builder as mbm_builder
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as partitioned_dataset
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.utilities as putil

from tarantella import logger
import tarantella as tnt
import tarantella.keras.utilities as utilities

import tensorflow as tf
from tensorflow.python.keras.saving.saved_model import json_utils
import json

try:
  import yaml
except ImportError:
  yaml = None

class CompileProperties:
  def __init__(self, model, params):
    # params = {'optimizer', 'loss', 'metrics', 'loss_weights',
    #           'sample_loss_weights', 'weighted_metrics'}
    self._optimizer = params.get('optimizer', None)
    self._loss = self._assign_named_attributes_to_outputs(model, 'loss', params)
    self._loss_weights = self._assign_named_attributes_to_outputs(model, 'loss_weights', params)
    self._metrics = self._assign_named_attributes_to_outputs(model, 'metrics', params)
    logger.debug(f"Compile properties: losses = {self._loss}, loss_weights = {self._loss_weights}, metrics = {self._metrics}")

  def _assign_named_attributes_to_outputs(self, model, attr_name, compile_params):
    attribute_list = compile_params[attr_name] if attr_name in compile_params.keys() else dict()

    if isinstance(attribute_list, dict):
      return attribute_list
    
    if not isinstance(attribute_list, list):
      attribute_list = [attribute_list]
    attr = dict()
    for index, out in enumerate(model.outputs):  
      attr[index] = attribute_list[index]
    return attr

  @property
  def loss(self):
    return self._loss

  @property
  def loss_weights(self):
    return self._loss_weights

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def metrics(self):
    return self._metrics


class PartitionedModel(parallel_model.ParallelModel):
  def __init__(self, model, group, partition_generator, rank_mapper,
               num_pipeline_stages = None):
    super().__init__(model = model, group = group)
    self._model_name = model.name
    self.built = False
    self.compile_properties = None
    self.num_pipeline_stages = num_pipeline_stages

    connection_table = rank_mapper.get_connections_for_rank(self.rank)
    self.pipeline_communicator = tnt.PipelineCommunicator(connection_table, self.num_pipeline_stages)
    self.initialized = False

    partition_id = rank_mapper.get_partition_for_rank(self.rank)
    partition_graph = partition_generator.get_partition_graph(partition_id)
    self.partition_info = pinfo.PartitionInfo(partition_id, partition_graph)

    core_model_builder = cm_builder.CoreModelBuilder(model, partition_id,
                                                     partition_graph)
    self.model = core_model_builder.get_model()
    self.nano_batch_size = None
    self.built = False


  @property
  def metrics(self):
    metrics_name_and_info = { m.name : m for m in self.model.metrics }
    user_defined_metrics = putil.extract_user_visible_metrics(metrics_name_and_info)
    return [v[0] for _,v in user_defined_metrics.items()]

  @property
  def metrics_names(self):
    metrics_name_and_info = { m.name : m for m in self.model.metrics }
    user_defined_metrics = putil.extract_user_visible_metrics(metrics_name_and_info)
    return [metric_name for metric_name in user_defined_metrics.keys()]

  @property
  def name(self):
    return self._model_name

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    self.built = True
    params = dict(locals())
    logger.info(f"[PartitionedModel] compile.")
    self.compile_properties = CompileProperties(self.model, params)
    return self.model.compile(optimizer,
                              loss,
                              metrics,
                              loss_weights,
                              sample_weight_mode,
                              weighted_metrics,
                              **kwargs)


  def evaluate(self,
               x = None,
               y = None,
               callbacks = None,
               **kwargs):
    self._configure_rebuild(dataset = x)
    self._build_model_and_compile_if_necessary()

    processed_callbacks = utilities._preprocess_pipelining_callbacks(callbacks, self.group,
                                                                     exec_type = 'evaluate',
                                                                     verbose = kwargs.get('verbose', None))

    ds = self._get_microbatched_dataset(dataset = x, nano_batch_size = self.nano_batch_size,
                                        num_pipeline_stages = self.num_pipeline_stages)
    test_loss_metrics = self.model.evaluate(x = ds, callbacks = processed_callbacks, **kwargs)
    user_visible_loss_metrics = putil.extract_user_visible_metrics(
                                      dict(zip(self.model.metrics_names, test_loss_metrics)))
    if len(user_visible_loss_metrics) == 1:
      return user_visible_loss_metrics[0]
    else:
      return [i for item in user_visible_loss_metrics.values() for i in item]

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
    logger.info(f"[PartitionedModel] fit.")
    self._configure_rebuild(dataset = x)
    self._build_model_and_compile_if_necessary()
    processed_callbacks = utilities._preprocess_pipelining_callbacks(callbacks, self.group,
                                                                     exec_type = 'fit',
                                                                     verbose = kwargs.get('verbose', None))

    ds = self._get_microbatched_dataset(dataset = x, nano_batch_size = self.nano_batch_size,
                                        num_pipeline_stages = self.num_pipeline_stages)
    return self.model.fit(x = ds, callbacks = processed_callbacks,
                          validation_data = validation_data,
                          **kwargs)

  def get_config(self):
    config = super().get_config()
    if 'name' in config.keys():
      config['name'] = self.name
    return config

  def get_weights(self):
    if not self.model.built:
      if not self.input_shapes:
        raise RuntimeError("""Cannot get weights before initializition.
        Please call "tnt.Model.build()" or "tnt.Model.fit()" first.
        """)
      self.model.build(self.input_shapes)
    return self.model.get_weights()

  def load_weights(self, filepath, **kwargs):
    return self.model.load_weights(filepath = filepath, **kwargs)

  def predict(self,
              x = None,
              callbacks = None,
              tnt_micro_batch_size = None,
              tnt_distribute_dataset = True,
              **kwargs):
    self._configure_rebuild(dataset = x)
    self._build_model_and_compile_if_necessary()

    processed_callbacks = utilities._preprocess_pipelining_callbacks(callbacks, self.group,
                                                                     exec_type = 'predict',
                                                                     verbose = kwargs.get('verbose', None))

    ds = self._get_microbatched_dataset(dataset = x, nano_batch_size = self.nano_batch_size,
                                        num_pipeline_stages = self.num_pipeline_stages)
    test_loss_metrics = self.model.predict(x = ds, callbacks = processed_callbacks, **kwargs)
    if tnt.is_group_master_rank(self.group):  # last partition
      return test_loss_metrics

  def save(self, filepath, tnt_save_all_devices = False, **kwargs):
    raise NotImplementedError("[PartitionedModel] `save` not supported")

  def save_weights(self, filepath, tnt_save_all_devices = False, **kwargs):
    raise NotImplementedError("[PartitionedModel] `save_weights` not supported")

  def set_weights(self, weights):
    self.model.set_weights(weights)

  def to_json(self, **kwargs):
    model_config = self.model._updated_config()
    model_config['config'] = self.get_config()
    return json.dumps(model_config, default=json_utils.get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    if yaml is None:
      raise ImportError(
          'Requires yaml module installed (`pip install pyyaml`).')
    model_config = self.model._updated_config()
    model_config['config'] = self.get_config()
    return yaml.dump(model_config, **kwargs)

  ####################
  # Helper functions #
  ####################

  def _get_microbatched_model_builder(self, micro_batch_size):
    if not self.initialized:
      self.pipeline_communicator.setup_infrastructure(micro_batch_size)
      self.initialized = True
    shared_model_builder = sm_builder.SharedModelBuilder(self.partition_info, self.model,
                                                         self.pipeline_communicator, micro_batch_size)
    shared_model = shared_model_builder.get_model()
    microbatched_model_builder = mbm_builder.MicrobatchedModelBuilder(self.partition_info,
                                                                      shared_model,
                                                                      micro_batch_size,
                                                                      self.num_pipeline_stages)
    return microbatched_model_builder

  def _get_microbatched_dataset(self, dataset, nano_batch_size, num_pipeline_stages):
    tnt_dataset = tnt.data.Dataset(dataset = dataset,
                                    num_ranks = 1,
                                    rank = 0)
    tnt_dataset.distribute_dataset_across_ranks(apply_batch = False)

    samples = tnt_dataset.base_dataset.map(lambda s,_: s)
    labels = tnt_dataset.base_dataset.map( lambda _,l: l)

    partition_samples = []
    partition_labels = []
    # assume all inputs are passed to the same start partition and
    # all outputs are generated on the same final partition
    if len(self.partition_info.get_real_ids(pinfo.EndpointDirection.inp)) > 0:
      partition_samples = [samples]
    if len(self.partition_info.get_real_ids(pinfo.EndpointDirection.out)) > 0:
      partition_labels = [labels]

    return partitioned_dataset.create_micro_batched_dataset(samples = partition_samples,
                                                            labels = partition_labels,
                                                            partition_info = self.partition_info,
                                                            num_micro_batches = num_pipeline_stages,
                                                            micro_batch_size = nano_batch_size,
                                                            dataset_size = tnt_dataset.number_samples)


  def _get_partition_compile_params(self):
    if not self.compile_properties:
      raise LogicError("[PipelinedModel] `model.fit` called before `model.compile`")

    logger.debug(f"[PartitionedModel] Compiled partitioned model with losses={self.compile_properties.loss}, "
                f"metrics = {self.compile_properties.metrics} {self.model.metrics}")
    return {'optimizer' : self.compile_properties.optimizer,
            'loss' : self.microbatched_model_builder.get_losses(self.compile_properties.loss),
            'loss_weights' : self.microbatched_model_builder.get_loss_weights(),
            'metrics' : self.microbatched_model_builder.get_metrics(self.compile_properties.metrics)}


  def _configure_rebuild(self, dataset):
    self.built = False
    dist_dataset = tnt.data.Dataset(dataset = dataset,
                                    num_ranks = 1,
                                    rank = 0)
    dist_dataset.distribute_dataset_across_ranks(apply_batch = False)

    # model is already built with the same `nano_batch_size`
    if self.nano_batch_size == dist_dataset.micro_batch_size // self.num_pipeline_stages:
      self.built = True
      return

    micro_batch_size = dist_dataset.micro_batch_size
    self.nano_batch_size = micro_batch_size // self.num_pipeline_stages
    if self.nano_batch_size * self.num_pipeline_stages != micro_batch_size:
      logger.warn(f"[PartitionedModel] The micro-batch size {self.micro_batch_size} is not a multiple of "
                  f" the number of pipeline stages ({self.num_pipeline_stages}); removing the remainder.")


  def _build_model_and_compile_if_necessary(self):
    if self.built:
      logger.info(f"[PartitionedModel] Model already built with nano_batch_size={self.nano_batch_size}")
      return

    logger.info(f"[PartitionedModel] Building pipelined model with nano_batch_size={self.nano_batch_size}")
    self.microbatched_model_builder = self._get_microbatched_model_builder(self.nano_batch_size)
    self.model = self.microbatched_model_builder.get_model()

    compile_parameters = self._get_partition_compile_params()
    self.model.compile(**compile_parameters)
    self.built = True


  def close(self):
    del self.model
    del self.pipeline_communicator
