import tarantella.keras.layers as tnt_layers
import tarantella.strategy.pipelining.model_builder as model_builder
import tarantella.strategy.pipelining.partition_info as pinfo

import tensorflow.keras as keras

class SharedModelBuilder(model_builder.ModelBuilder):
  
  def __init__(self, partition_info, core_model, pipeline_communicator, micro_batch_size):
    super().__init__(partition_info)
    self.core_model = core_model
    self.pipeline_communicator = pipeline_communicator
    self.micro_batch_size = micro_batch_size

  def get_model(self):
    """ Add RecvLayers to all inputs that represent incoming connections
        Add SendLayers to all outputs that represent outgoing connections
        Add one 'sequential' placeholder input and output, respectively,
        to the model such that it can be shared by multiple micro_batches
        processed sequentially
    """
    # create inputs
    real_inputs = self.build_inputs(pinfo.EndpointType.inp)
    edge_inputs = self.build_inputs(pinfo.EndpointType.inp_edge)
    recv_tag_inputs = self.build_tag_inputs(pinfo.EndpointDirection.inp)
    send_tag_inputs = self.build_tag_inputs(pinfo.EndpointDirection.out)
    seq_input = self.build_seq_input()  # used to model sequential dependencies
  
    inputs = self.merge_inputs(real_inputs = real_inputs, edge_inputs = edge_inputs,
                               recv_tags = recv_tag_inputs, send_tags = send_tag_inputs,
                               seq_input = seq_input)
        
    # pass all core inputs through a `RemoveSeqInput` layer
    initial_inputs = self.merge_endpoints(real_endpoints = real_inputs,
                                          edge_endpoints = edge_inputs,
                                          seq_endpoint = seq_input)
    real_and_recv_inputs = tnt_layers.RemoveSeqInput()(initial_inputs)

    # add `RecvLayer`s to inputs representing incoming edges
    real_inputs, recv_inputs = self.split_endpoints_list(real_and_recv_inputs,
                                                         pinfo.EndpointDirection.inp)
    received_inputs = self.build_layers_for_receive_inputs(recv_inputs, recv_tag_inputs)
    inputs_to_core = self.merge_endpoints(real_endpoints = real_inputs,
                                          edge_endpoints = received_inputs)

    # execute core model on the pre-processed inputs
    outputs_from_core = self.core_model(inputs_to_core)

    # add `SendLayer`s to outputs representing outgoing edges
    real_outputs, send_outputs = self.split_endpoints_list(outputs_from_core,
                                                           pinfo.EndpointDirection.out)
    sent_outputs = self.build_layers_for_send_conections(send_outputs, send_tag_inputs)
    outputs = self.merge_endpoints(real_endpoints = real_outputs,
                                   edge_endpoints = sent_outputs)

    # pass all outputs through an `AddSeqOutput` layer
    outputs = tnt_layers.AddSeqOutput(micro_batch_size = self.micro_batch_size)(outputs)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name=f"p_{self.partition_info.pid}_{self.core_model.name}_shared")



  # create recv layers and return a list of RecvLayer outputs, which have to be fed to
  # the core model, together with additional `real` Input layers
  def build_layers_for_receive_inputs(self, recv_inputs, recv_tags_inputs):
    assert len(recv_inputs) == len(recv_tags_inputs)
    recv_outputs = []
    for index, input_id in enumerate(recv_inputs):
      recv_outputs.append(tnt_layers.RecvLayer(self.pipeline_communicator)(
                                              recv_inputs[index],
                                              recv_tags_inputs[index]))
    return recv_outputs


  # create send layers and return a list of SendLayer outputs to be fed into an `AddSeq` layer
  def build_layers_for_send_conections(self, send_inputs, send_tags_inputs):
    assert len(send_inputs) == len(send_tags_inputs)

    send_layer_outputs = []
    for index, _ in enumerate(send_inputs):
      send_layer_outputs.append(tnt_layers.SendLayer(self.pipeline_communicator)(
                                              send_inputs[index],
                                              send_tags_inputs[index]))
    return send_layer_outputs
