import tarantella.strategy.pipelining.partition_generator as pgen
import tarantella.strategy.pipelining.rank_mapper as rmapper
import tarantella.strategy.pipelining.core_model_builder as cm_builder
import tarantella.strategy.pipelining.shared_model_builder as sm_builder
import tarantella.strategy.pipelining.microbatched_model_builder as mbm_builder
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as partitioned_dataset
import tarantella.strategy.pipelining.partition_info as pinfo

from tarantella import logger
import tarantella as tnt

import tensorflow as tf

class CompileProperties:
  def __init__(self, model, params):
    # args_names = {'optimizer', 'loss', 'metrics', 'loss_weights',
    #               'sample_loss_weights', 'weighted_metrics'}
    logger.warn(params['optimizer'])
    self._optimizer = params['optimizer'] #if 'optimizer' in params.keys() else None
    self._loss = [params['loss']] #self._assign_named_attributes_to_outputs(model, 'loss', params)
    self._metrics = params['metrics'] #self._assign_named_attributes_to_outputs(model, 'metrics', params)

  def _assign_named_attributes_to_outputs(self, model, attr_name, compile_params):
    attribute_list = compile_params[attr_name] if attr_name in compile_params.keys() else dict()

    if isinstance(attribute_list, dict):
      return attribute_list
    
    if not isinstance(attribute_list, list):
      attribute_list = [attribute_list]
    attr = dict()
    for index, out in enumerate(model.outputs):  
      attr[out.name] = attribute_list[index]
    return attr

  @property
  def loss(self):
    return self._loss

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def metrics(self):
    return self._metrics


def to_microbatched(model, micro_batch_size, num_micro_batches, num_batches, num_test_batches):
  rank = tnt.get_rank()
  partition_generator = pgen.GraphPartitionGenerator(model)
  rank_mapper = rmapper.RankMapper(partition_generator.get_pipeline_graph(), tnt.get_size())
  core_model_builder = cm_builder.CoreModelBuilder(model, partition_generator,
                                                    rank_mapper, rank)
  core_model = core_model_builder.get_model()

  connection_table = rank_mapper.get_connections_for_rank(rank)
  pipeline_communicator = tnt.PipelineCommunicator(connection_table, micro_batch_size, num_micro_batches)

  partition_id = rank_mapper.get_partition_for_rank(rank)
  partition_info = pinfo.PartitionInfo(partition_id = partition_id,
                                       partition_graph = partition_generator.get_partition_graph(partition_id))

  shared_model_builder = sm_builder.SharedModelBuilder(partition_info, core_model,
                                                    pipeline_communicator, micro_batch_size)
  shared_model = shared_model_builder.get_model()

  microbatched_model_builder = mbm_builder.MicrobatchedModelBuilder(partition_info, shared_model,
                                                                     micro_batch_size, num_micro_batches)
  ds = load_microbatched_datasets(micro_batch_size, num_micro_batches,
                                  num_batches, num_test_batches, partition_info)
  return microbatched_model_builder, ds


class PartitionedModel(tf.keras.models.Model):
  def __init__(self, model, group, partition_generator, rank_mapper,
               num_pipeline_stages = None):
    super().__init__()
    self.compile_properties = None
    self.rank = group.rank
    self.num_pipeline_stages = num_pipeline_stages
    self.connection_table = rank_mapper.get_connections_for_rank(self.rank)
    
    partition_id = rank_mapper.get_partition_for_rank(self.rank)
    partition_graph = partition_generator.get_partition_graph(partition_id)
    self.partition_info = pinfo.PartitionInfo(partition_id, partition_graph)

    core_model_builder = cm_builder.CoreModelBuilder(model, partition_id,
                                                     partition_graph)
    self.model = core_model_builder.get_model()


  def _get_microbatched_model_builder(self, micro_batch_size):
    pipeline_communicator = tnt.PipelineCommunicator(self.connection_table,
                                                     micro_batch_size,
                                                     self.num_pipeline_stages)

    shared_model_builder = sm_builder.SharedModelBuilder(self.partition_info, self.model,
                                                                   pipeline_communicator, micro_batch_size)
    shared_model = shared_model_builder.get_model()
    microbatched_model_builder = mbm_builder.MicrobatchedModelBuilder(self.partition_info,
                                                                       shared_model,
                                                                       micro_batch_size,
                                                                       self.num_pipeline_stages)
    return microbatched_model_builder

  def _get_microbatched_dataset(self, dataset,
                                micro_batch_size, num_micro_batches):
    dist_dataset = tnt.data.Dataset(dataset = dataset,
                                    num_ranks = 1,
                                    rank = 0)
    print(tf.data.Dataset.zip(dist_dataset.base_dataset))
    samples, lables = iter(dist_dataset.base_dataset).next

    partition_samples = []
    partition_labels = []
    # assume all inputs are passed to the same start partition and
    # all outputs are generated on the same final partition
    if len(self.partition_info.get_real_ids(pinfo.EndpointDirection.inp)) > 0:
      partition_samples = [tf.data.Dataset.from_tensor_slices(samples)]
    if len(self.partition_info.get_real_ids(pinfo.EndpointDirection.out)) > 0:
      partition_labels = [tf.data.Dataset.from_tensor_slices(labels)]

    return partitioned_dataset.create_micro_batched_dataset(samples = partition_samples,
                                                  labels = partition_labels,
                                                  partition_info = self.partition_info,
                                                  num_micro_batches = num_micro_batches,
                                                  micro_batch_size = micro_batch_size)


  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    params = dict(locals())
    logger.warn("[PartitionedModel] compile")
    self.compile_properties = CompileProperties(self.model, params)
    return self.model.compile(optimizer='rmsprop',
                                   loss=None,
                                   metrics=None,
                                   loss_weights=None,
                                   sample_weight_mode=None,
                                   weighted_metrics=None,
                                   **kwargs)


  def _get_partition_compile_params(self):
    if not self.compile_properties:
      raise LogicError("[PipelinedModel] `model.fit` called before `model.compile`")

    return {'optimizer' : self.compile_properties.optimizer,
            'loss' : self.microbatched_model_builder.get_losses(self.compile_properties.loss),
            'loss_weights' : self.microbatched_model_builder.get_loss_weights(),
            'metrics' : self.microbatched_model_builder.get_metrics(self.compile_properties.metrics)}



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
    micro_batch_size = 2
    
    self.microbatched_model_builder = self._get_microbatched_model_builder(micro_batch_size)
    self.model = self.microbatched_model_builder.get_model()
    
    compile_parameters = self._get_partition_compile_params()
    self.model.compile(**compile_parameters)

    ds = self._get_microbatched_dataset(x, micro_batch_size, num_micro_batches = self.num_pipeline_stages)
    return self.model.fit(ds, callbacks, validation_data, **kwargs)

  @property
  def distribute_strategy(self):
    return tf.distribute.get_strategy()

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

  def get_config(self):
    return self.model.get_config()

  def get_layer(self, name=None, index=None):
    return self.model.get_layer(name, index)
  
  def get_weights(self):
    return self.model.get_weights()

  def load_weights(self, filepath, **kwargs):
    return self.model.load_weights(filepath = filepath, **kwargs)
