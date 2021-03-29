import tarantella.keras.layers as tnt_layers
import tarantella.strategy.pipelining.model_builder as model_builder
import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.pipeline_microbatched_dataset as dataset_utils

import tensorflow.keras as keras

class MicrobatchedModelBuilder(model_builder.ModelBuilder):
  def __init__(self, partition_info, shared_model, 
               micro_batch_size, number_micro_batches):
    super().__init__(partition_info)
    self.partition_info = partition_info
    self.shared_model = shared_model
    self.micro_batch_size = micro_batch_size
    self.number_micro_batches = number_micro_batches

  def get_model(self):
    # create inputs
    real_inputs_by_mbatch = self.build_microbatched_inputs(pinfo.EndpointType.inp)
    edge_inputs_by_mbatch = self.build_microbatched_inputs(pinfo.EndpointType.inp_edge)

    recv_tags_by_mbatch = self.build_microbatched_tag_inputs(pinfo.EndpointDirection.inp)
    send_tags_by_mbatch = self.build_microbatched_tag_inputs(pinfo.EndpointDirection.out)

    seq_input = self.build_seq_input()

    # roll out microbatched shared models (slice inputs and plug them to corresponding sub-models)
    current_seq_input = seq_input
    outputs_by_mbatch = {}
    for mbatch_id in range(self.number_micro_batches):
      shared_model_inputs = self.merge_inputs(real_inputs = real_inputs_by_mbatch[mbatch_id],
                                              edge_inputs = edge_inputs_by_mbatch[mbatch_id],
                                              recv_tags = recv_tags_by_mbatch[mbatch_id],
                                              send_tags = send_tags_by_mbatch[mbatch_id],
                                              seq_input = current_seq_input)

      stage_outputs = self.shared_model(shared_model_inputs)
      
      # the last output will be used as the `sequential` input of the next micro_batch
      current_seq_input = [stage_outputs[-1]]
      # the remaining outputs represent the list of outputs of the current micro_batch
      outputs_by_mbatch[mbatch_id] = stage_outputs[:-1]
    seq_output = current_seq_input

    real_outputs_by_mbatch, edge_outputs_by_mbatch = self.slice_microbatched_outputs(outputs_by_mbatch)

    # create identity layers to rename all types of outputs
    real_outputs_by_mbatch = self.rename_microbatched_outputs(real_outputs_by_mbatch, pinfo.EndpointType.out)
    edge_outputs_by_mbatch = self.rename_microbatched_outputs(edge_outputs_by_mbatch, pinfo.EndpointType.out_edge)
    seq_output = self.rename_seq_output(seq_output)

    inputs = self.build_microbatched_inputs_list(real_inputs_by_mbatch = real_inputs_by_mbatch,
                                                 edge_inputs_by_mbatch = edge_inputs_by_mbatch,
                                                 recv_tags_by_mbatch = recv_tags_by_mbatch,
                                                 send_tags_by_mbatch = send_tags_by_mbatch,
                                                 seq_input = seq_input)
    outputs = self.build_microbatched_outputs_list(real_outputs_by_mbatch = real_outputs_by_mbatch,
                                                   edge_outputs_by_mbatch = edge_outputs_by_mbatch,
                                                   seq_output = seq_output)
    return keras.Model(inputs=inputs, outputs=outputs,
                       name=f"p_{self.partition_info.pid}_{self.shared_model.name}_microbatched")


  def slice_microbatched_outputs(self, outputs_by_mbatch):
    real_outputs_by_mbatch = {}
    edge_outputs_by_mbatch = {}
    for mbatch_id in range(self.number_micro_batches):
      real, edge = self.split_endpoints_list(outputs_by_mbatch[mbatch_id], pinfo.EndpointDirection.out)
      real_outputs_by_mbatch[mbatch_id] = real
      edge_outputs_by_mbatch[mbatch_id] = edge
    return real_outputs_by_mbatch, edge_outputs_by_mbatch

  def build_microbatched_outputs_list(self, real_outputs_by_mbatch, edge_outputs_by_mbatch, seq_output):
    # assemble list of final outputs for the microbatched model
    outputs = []
    for mbatch_id in range(self.number_micro_batches):
      mbatch_outputs = self.merge_endpoints(real_endpoints = real_outputs_by_mbatch[mbatch_id],
                                            edge_endpoints = edge_outputs_by_mbatch[mbatch_id])
      outputs += mbatch_outputs
    outputs += seq_output
    return outputs

  def build_microbatched_inputs(self, endpoint_type):
    input_infos = self.partition_info.get_infos(endpoint_type)

    micro_batched_inputs = {}
    for mbatch_id in range(self.number_micro_batches):
      micro_batched_inputs[mbatch_id] = []
      for index, info in enumerate(input_infos):
        input_name = dataset_utils.create_name_micro_batched_layer(self.partition_info.pid,
                                    element_type = endpoint_type,
                                    layer_id = index, # index within the real inputs
                                    micro_batch_id = mbatch_id)
        micro_batched_inputs[mbatch_id] += [keras.Input(shape = info.shape[1:], dtype = info.dtype,
                                                        name=input_name)]
    return micro_batched_inputs

  def build_microbatched_tag_inputs(self, endpoint_direction):
    tag_inputs_by_mbatch = {}
    for mbatch_id in range(self.number_micro_batches):
      tag_inputs_by_mbatch[mbatch_id] = self.build_tag_inputs(endpoint_direction, mbatch_id)
    return tag_inputs_by_mbatch

  def rename_microbatched_outputs(self, outputs_by_mbatch, output_type):  
    renamed_outputs_by_mbatch = dict()
    for mbatch_id in range(self.number_micro_batches):
      renamed_outputs_by_mbatch[mbatch_id] = list()
      for index, out in enumerate(outputs_by_mbatch[mbatch_id]):
        layer = self.rename_by_identity_layer(layer = out,
                                              element_type = output_type,
                                              layer_id = index, mbatch_id = mbatch_id)
        renamed_outputs_by_mbatch[mbatch_id] += [layer]
    return renamed_outputs_by_mbatch

  def rename_seq_output(self, seq_output):  
    return self.rename_by_identity_layer(layer = seq_output,
                                         element_type = pinfo.EndpointType.seq_output)

  def rename_by_identity_layer(self, layer, element_type, layer_id = None, mbatch_id = None):
    name = dataset_utils.create_name_micro_batched_layer(self.partition_info.pid,
                                                         element_type = element_type,
                                                         layer_id = layer_id,
                                                         micro_batch_id = mbatch_id)
    return tnt_layers.IdentityLayer(name=name)(layer)
