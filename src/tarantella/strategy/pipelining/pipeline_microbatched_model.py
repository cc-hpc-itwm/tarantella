import os
import copy
from enum import Enum
import numpy as np
import string

import communication_layers as comm_layers
import utilities as utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_index_partition_input_outputs(layer_id, micro_batch_id, num_layers, num_micro_batches):
  return num_layers * micro_batch_id + layer_id

class PartitionInfo:
  def __init__(self, num_inputs, num_outputs):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs

class CommunicationInfo:
  def __init__(self, src_partition = None, dest_partition = None, 
           src_output_index = None, dest_input_index = None):
    self.src_partition = src_partition
    self.dest_partition = dest_partition
    self.src_output_index = src_output_index
    self.dest_input_index = dest_input_index

class NoCommunication(CommunicationInfo):
  def __init__(self):
    super(NoCommunication).__init__()

def add_sequential_input_output(model, name, micro_batch_size):
  """ Add one 'sequential' placeholder input and output, respectively,
      to an existing model such that it can be shared by multiple micro_batches
      processed sequentially
  """
  inputs = [keras.Input(tensor = inp) for inp in model.inputs]
  input_seq = keras.Input(shape=(1,))  # use to model sequential dependencies (only one needed)
  inputs = inputs + [input_seq] # sequential input must be last
  x = comm_layers.DropLastInput()(inputs)
  x = model(x)
  outputs = comm_layers.AddFakeOutput(size_fake_output=micro_batch_size)(x)
  return keras.Model(inputs=inputs, outputs=outputs, name=name)

def get_communication_tags(partition_table, num_micro_batches, communication_info, micro_batch_id):
  """ Generate communication tags for input Communication layers/output Communication losses
      Assumptions:
      - tags are always generated relative to the source (sending) partition
      - order of generated tags: [src_partition_id][micro_batch_id][src_output_id][fwd/bwd]
  """
  src_partition_id = communication_info.src_partition
  used_tags = 0
  for part_id in range(src_partition_id):
    used_tags += num_micro_batches * partition_table[part_id].num_outputs * 2 

  num_outputs = partition_table[src_partition_id].num_outputs
  fwd_tag = used_tags + micro_batch_id * num_outputs * 2 + communication_info.src_output_index * 2
  bwd_tag = fwd_tag + 1
  return fwd_tag, bwd_tag

class LayerType(Enum):
    input = 'input'
    output = 'output'
    start_seq = 'start_seq'
    end_seq = 'end_seq'

class MicrobatchedModelGenerator:
  def __init__(self, core_model, micro_batch_size, num_micro_batches, partition_id, 
               context, partition_table, input_info, output_info):
    if not isinstance(input_info, dict):
      raise TypeError("[MicrobatchedModelGenerator] `input_info` should be a dictionary")
    if len(input_info) != len(core_model.inputs):
      raise ValueError("[MicrobatchedModelGenerator] incorrect `input_info` length")
    if not isinstance(output_info, dict):
      raise TypeError("[MicrobatchedModelGenerator] `output_info` should be a dictionary")
    if len(output_info) != len(core_model.outputs):
      raise ValueError("[MicrobatchedModelGenerator] incorrect `output_info` length")
    
    self.shared_model = add_sequential_input_output(core_model,
                                                    utils.create_name_partition(partition_id) + "_shared",
                                                    micro_batch_size)
    self.micro_batch_size = micro_batch_size
    self.num_micro_batches = num_micro_batches
    self.partition_id = partition_id
    self.partition_table = partition_table
    self.input_info = input_info
    self.output_info = output_info
    self.num_core_inputs = len(core_model.inputs)
    self.num_core_outputs = len(core_model.outputs)
    self.context = context

  def get_model(self):
    list_inputs, list_received_inputs = self.__create_micro_batched_inputs_and_comm_layers()
    list_outputs_temp = self.__replicate_for_micro_batches(list_received_inputs)
    list_outputs = self.__rename_outputs(list_outputs_temp)
    model = MicrobatchedModel(inputs=list_inputs,
                              outputs=list_outputs,
                              name=utils.create_name_partition(self.partition_id),
                              num_micro_batches=self.num_micro_batches,
                              last_partition = (self.partition_id == 1))
    return model
    
  # TODO: check whether can be refactored
  def __create_micro_batched_inputs_and_comm_layers(self):
    ''' Create list of `micro_batched_inputs` by replicating the `core_inputs` by 
        the number of micro_batches and adding a sequential layer at the end.
        The layout of the `micro_batched_inputs` list is:
            [mb_0_input_0, mb_0_input_1, mb_1_input_0, mb_1_input_1].
        Build `comm_layers` for inputs that require communication and create a 
        `received_inputs` list containing either a micro_batched input, or the output 
        of the corresponding `comm_layer`.
    '''
    num_micro_batched_inputs = self.num_micro_batches * self.num_core_inputs + 1
    micro_batched_inputs = [None] * num_micro_batched_inputs
    received_inputs = [None] * num_micro_batched_inputs

    for mbatch_id in range(self.num_micro_batches):
      for layer_id in range(self.num_core_inputs):
        input_name = utils.create_name_micro_batched_layer(partition_id = self.partition_id, 
                                                          micro_batch_id = mbatch_id,
                                                          layer_type = utils.InoutLayerType.input,
                                                          layer_id = layer_id)
        input_shape = self.shared_model.inputs[layer_id].shape[1:] # input tensors must not include batch size

        index = self.__get_index_partition_inputs(layer_id, mbatch_id)
        micro_batched_inputs[index] = keras.Input(shape=input_shape, name=input_name)

        comm_info = self.input_info[layer_id]
        if isinstance(comm_info, NoCommunication):
          received_inputs[index] = micro_batched_inputs[index]
        else:
          comm_layer = self.__create_comm_layer(comm_info, layer_id, mbatch_id)
          received_inputs[index] = comm_layer(micro_batched_inputs[index])

    input_name_seq = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                           layer_type = utils.InoutLayerType.start_seq)
    micro_batched_inputs[-1] = keras.Input(shape=(1,), name=input_name_seq)
    received_inputs[-1] = micro_batched_inputs[-1]
    return micro_batched_inputs, received_inputs

  def __create_comm_layer(self, communication_info, layer_id, micro_batch_id):
    fwd_tag, bwd_tag = get_communication_tags(partition_table = self.partition_table,
                                              num_micro_batches = self.num_micro_batches,
                                              communication_info = communication_info,
                                              micro_batch_id = micro_batch_id)
    input_shape = self.shared_model.inputs[layer_id].shape
    comm_layer = comm_layers.CommLayer(self.context, src_dest=communication_info.src_partition,
                                       fwd_tag=fwd_tag,
                                       bwd_tag=bwd_tag,
                                       output_shape=input_shape)
    return comm_layer

  def __create_comm_loss(self, communication_info, layer_id, micro_batch_id):
    fwd_tag, bwd_tag = get_communication_tags(partition_table = self.partition_table,
                                              num_micro_batches = self.num_micro_batches,
                                              communication_info = communication_info,
                                              micro_batch_id = micro_batch_id)
    comm_loss = comm_layers.CommLoss(self.context,
                                     src_dest=communication_info.src_partition,
                                     fwd_tag=fwd_tag,
                                     bwd_tag=bwd_tag)
    return comm_loss

  def __replicate_for_micro_batches(self, inputs):
    ''' Create micro_batched `outputs` by sequentially executing the `shared_model` 
        for each micro_batch
    '''
    outputs = list()
    seq_token = inputs[-1]
    for mbatch_id in range(self.num_micro_batches):
      inputs_id_start = self.num_core_inputs * mbatch_id
      inputs_id_end = inputs_id_start + self.num_core_inputs
      stage_outputs = self.shared_model(inputs[inputs_id_start:inputs_id_end] + [seq_token])
      # the last output will be used as the `sequential` input of the next micro_batch
      seq_token = stage_outputs[-1]
      # the remaining outputs are appended to the list of outputs of the micro_batched model
      outputs += stage_outputs[:-1]
    outputs += [stage_outputs[-1]]
    return outputs
   
  def __rename_outputs(self, micro_batched_outputs):  
    outputs = [None] * len(micro_batched_outputs)
    for layer_id in range(self.num_core_outputs):
      for mbatch_id in range(self.num_micro_batches):
        name = utils.create_name_micro_batched_layer(self.partition_id,
                                                     utils.InoutLayerType.output,
                                                     layer_id, mbatch_id)
        index = self.__get_index_partition_outputs(layer_id, mbatch_id)
        outputs[index] = comm_layers.IdentityLayer(name=name)(micro_batched_outputs[index])

    seq_name = utils.create_name_micro_batched_layer(self.partition_id,
                                                     utils.InoutLayerType.end_seq)
    outputs[-1] = comm_layers.IdentityLayer(name=seq_name)(micro_batched_outputs[-1])
    return outputs

  def __get_index_partition_inputs(self, layer_id, micro_batch_id):
    return get_index_partition_input_outputs(layer_id = layer_id,
                                             micro_batch_id = micro_batch_id, 
                                             num_layers = self.num_core_inputs,
                                             num_micro_batches = self.num_micro_batches)

  def __get_index_partition_outputs(self, layer_id, micro_batch_id):
    return get_index_partition_input_outputs(layer_id = layer_id,
                                             micro_batch_id = micro_batch_id, 
                                             num_layers = self.num_core_outputs,
                                             num_micro_batches = self.num_micro_batches)

  def get_losses(self, core_losses):
    ''' Create list of losses for microbatched models:
        * duplicate user losses for each micro_batch
        * build `comm_loss`es for each communication output and micro_batch
        * add fake loss (`ZeroLoss`) for last seqential output
    '''
    losses = dict()
    for output_id, comm_info in self.output_info.items():
      for mbatch_id in range(self.num_micro_batches):
        name = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                     layer_type = utils.InoutLayerType.output,
                                                     layer_id = output_id,
                                                     micro_batch_id = mbatch_id)
        if isinstance(comm_info, NoCommunication): # real loss
          loss = core_losses[output_id]
          losses[name] = copy.deepcopy(loss)
        else: # CommLoss
          comm_loss = self.__create_comm_loss(comm_info, output_id, mbatch_id)
          losses[name] = comm_loss
    name_seq = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                     layer_type = utils.InoutLayerType.end_seq)
    losses[name_seq] = comm_layers.ZeroLoss()
    return losses

  def get_metrics(self, core_metrics):
    ''' Create list of metrics for microbatched models:
        * duplicate user metrics for each micro_batch (if provided)
        * add fake metrics (`ZeroMetric`) for each not provided user metric and micro_batch
        * add fake metric (`ZeroMetric`) for last seqential output, if necessary

        Assumption:
        Keras requires that the keys in loss and metric dicts are identical.
    '''
    if not core_metrics:
      return None

    metrics = dict()
    for output_id, comm_info in self.output_info.items():
      for mbatch_id in range(self.num_micro_batches):
        if isinstance(comm_info, NoCommunication): # real metric
          name = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                       layer_type = utils.InoutLayerType.output,
                                                       layer_id = output_id,
                                                       micro_batch_id = mbatch_id)
          if output_id in core_metrics:
            metric = core_metrics[output_id]
          else:
            metric = comm_layers.ZeroMetric()
          metrics[name] = copy.deepcopy(metric)

    if metrics: # not empty
      name_seq = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                       layer_type = utils.InoutLayerType.end_seq)
      metrics[name_seq] = comm_layers.ZeroMetric()
    return metrics

  def get_loss_weights(self, core_loss_weights):
    ''' Create list of loss weights (scalings) for microbatched models:
        * duplicate user weights for each micro_batch (if provided),
          and rescale by 1. / `num_micro_batches`
          (in order to ensure correct averaging over entire mini-batch in the loss)
        * add zeros for each not provided user weight and micro_batch
        * add zero loss weight for last seqential output
    '''
    loss_weights = dict()
    for output_id, comm_info in self.output_info.items():
      for mbatch_id in range(self.num_micro_batches):
        name = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                     layer_type = utils.InoutLayerType.output,
                                                     layer_id = output_id,
                                                     micro_batch_id = mbatch_id)
        if isinstance(comm_info, NoCommunication): # real loss
          loss_weights[name] = 1. / self.num_micro_batches
          if output_id in core_loss_weights: # adjust for user loss weight
            loss_weights[name] *= core_loss_weights[output_id]
        else: # CommLoss
          loss_weights[name] = 0.
    name_seq = utils.create_name_micro_batched_layer(partition_id = self.partition_id,
                                                     layer_type = utils.InoutLayerType.end_seq)
    loss_weights[name_seq] = 0.
    return loss_weights

  def get_train_dataset(self):
    pass

  def get_val_dataset(self):
    pass

  def get_test_dataset(self):
    pass

# TODO only do the print stuff here
class MicrobatchedModel (keras.Model):
  '''
  TODO: describe:
        - adds losses/metrics/loss_weights for micro_batches
        - correct printing of loss/metrics
        - add parameter to switch to full printing of all losses
  '''
  def __init__(self, *args, num_micro_batches, last_partition = False, **kwargs):
    super(MicrobatchedModel, self).__init__(*args, **kwargs)
    self.num_micro_batches = num_micro_batches
    self.last_partition = last_partition
  
  def extract_final_metrics(self, losses_and_metrics):
    """ Assume default metrics in train/test steps include a loss and an average metric value 
        (over micro_batches) for all other metrics defined.
        
        Assume that the names of micro-batch metrics include the string `micro` 
        and the fake (sequential) metrics include `seq`.
        """
    assert 'loss' in losses_and_metrics, """[extract_final_metrics] Computed metrics after 
                                            train/test step do not have a `loss` element"""
    final_metrics = dict()
    for m in losses_and_metrics.keys():
      if not 'micro' in m and not 'seq' in m:
        final_metrics[m] = losses_and_metrics[m]
    return final_metrics

  def train_step(self, data):
    """ The logic for one training step.
        Remove micro-batch-level losses and metrics from the default output"""
    results = super(MicrobatchedModel, self).train_step(data)
    return self.extract_final_metrics(results)

  def test_step(self, data):
    """ The logic for one test step.
        Remove micro-batch-level losses and metrics from the default output"""
    results = super(MicrobatchedModel, self).test_step(data)
    return self.extract_final_metrics(results)

  def evaluate(self, *args, **kwargs):
    """ Returns the loss value & metrics values for the model in test mode."""
    results = super(MicrobatchedModel, self).evaluate(*args, **kwargs)
    if isinstance(results, dict):
      return results
    if isinstance(results, list):
      return [x for x in results if x is not None]
    return results