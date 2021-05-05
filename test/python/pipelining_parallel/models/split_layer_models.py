import tarantella.strategy.pipelining.partition_info as pinfo
import tarantella.strategy.pipelining.partition_generator as pgen
import utilities as util

import tensorflow as tf
import tensorflow.keras as keras

p_0_rank = 0
p_1_rank = 1
p_2_rank = 2

def alexnet_model_generator():
  util.set_tf_random_seed()
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), name='conv')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='maxpool')(x)
  x = pgen.SplitLayer(name="split_layer0")(x)

  x = keras.layers.Conv2D(32, 3, strides=(1, 1), name='conv_two')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='maxpool_two')(x)
  x = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(64, 3, strides=(1, 1), name='conv_three')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_three')(x)
  x = keras.layers.Flatten(name = 'flatten')(x)
  x = keras.layers.Dense(512, activation='relu', name = 'dense_relu')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name = 'dense_softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def alexnet_partition_info(ref_model, rank):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo('0')

    in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]

    out_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo('1')
    in_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0]

    out_edge_0 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0]

  elif rank == p_2_rank:
    partition_info = pinfo.PartitionInfo('2')
    in_edge_0 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0]

    out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = [out_0]
  return partition_info

def alexnet_partitioned_core_model(rank):
  # --- core model on partition 0
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), name='conv')(input0)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name = 'maxpool')(x)
  x = keras.layers.Layer(name="split_layer0_input")(x)
  core_model0 = keras.Model(inputs=input0, outputs=x)

  # --- core model on partition 1
  input1 = keras.Input(shape=(24,24,32,), name='split_layer0_output')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), name='conv_two')(input1)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='maxpool_two')(x)
  x = keras.layers.Layer(name="split_layer1_input")(x)
  core_model1 = keras.Model(inputs=input1, outputs=x)

  # --- core model on partition 2
  input2 = keras.Input(shape=(20,20,32,), name='split_layer1_output')
  x = keras.layers.Conv2D(64, 3, strides=(1, 1), name='conv_three')(input2)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_three')(x)
  x = keras.layers.Flatten(name = 'flatten')(x)
  x = keras.layers.Dense(512, activation='relu', name = 'dense_relu')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name = 'dense_softmax')(x)
  core_model2 = keras.Model(inputs=input2, outputs=outputs)

  if rank == p_0_rank:
    return core_model0
  elif rank == p_1_rank:
    return core_model1
  elif rank == p_2_rank:
    return core_model2


def fc_model_generator():
  util.set_tf_random_seed()
  reference_input = keras.Input(shape=(28,28,1,), name='reference_input')
  reference_x = keras.layers.Flatten(name = 'flatten')(reference_input)
  reference_x = keras.layers.Dense(10, activation='relu', name='dense_relu')(reference_x)
  reference_x = pgen.SplitLayer(name="split_layer1")(reference_x)
  reference_output = keras.layers.Dense(10,
                                  activation='softmax',
                                  name='dense_softmax')(reference_x)
  reference_model = keras.Model(inputs=reference_input, outputs=reference_output, name="reference_model")
  return reference_model

def fc_partition_info(ref_model, rank):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo('0')

    in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]
    out_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo('1')
    in_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0]

    out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = [out_0]
  return partition_info


def fc_partitioned_core_model(rank):
  # --- core model on partition 0
  util.set_tf_random_seed()
  reference_input = keras.Input(shape=(28,28,1,), name='reference_input')
  reference_x = keras.layers.Flatten(name = 'flatten')(reference_input)
  reference_x = keras.layers.Dense(10, activation='relu', name='dense_relu')(reference_x)
  reference_x = keras.layers.Layer(name="split_layer1_input")(reference_x)
  p0_model = keras.Model(inputs=reference_input, outputs=reference_x, name="p_0")

  # --- core model on partition 1
  input0 = keras.Input(shape=(10,), name='split_layer1_output')
  output = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(input0)
  p1_model = keras.Model(inputs=input0, outputs=output, name="p_1")

  if rank == p_0_rank:
    return p0_model
  elif rank == p_1_rank:
    return p1_model

def skip_connection_model_generator():
  util.set_tf_random_seed()
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', name='conv')(inputs)
  y = pgen.SplitLayer(name="split_layer0")(x)
  z = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', activation='relu',
                          name='conv_relu')(y)
  x = pgen.SplitLayer(name="split_layer2")(x)

  x = keras.layers.Concatenate(name='concat')([x, z])
  x = keras.layers.Flatten(name='flatten')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def skip_connection_partition_info(ref_model, rank):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo('0')
    in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]

    out_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    out_edge_1 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0, out_edge_1]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo('1')
    in_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0]

    out_edge_0 = pinfo.EndpointInfo(2, ref_model.get_layer('split_layer2').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0]

  elif rank == p_2_rank:
    partition_info = pinfo.PartitionInfo('2')
    in_edge_0 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    in_edge_1 = pinfo.EndpointInfo(2, ref_model.get_layer('split_layer2').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0, in_edge_1]

    out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = [out_0]
  return partition_info

def skip_connection_partitioned_core_model(rank):
  util.set_tf_random_seed()
  # --- core model on partition 0
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', name='conv')(input0)
  y = keras.layers.Layer(name="split_layer0_input")(x)
  z = keras.layers.Layer(name="split_layer1_input")(x)
  core_model0 = keras.Model(inputs=input0, outputs=[y, z])

  # --- core model on partition 1
  input1 = keras.Input(shape=(26,26,32,), name='split_layer0_output')
  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', activation='relu',
                          name='conv_relu')(input1)
  x = keras.layers.Layer(name="split_layer2_input")(x)
  core_model1 = keras.Model(inputs=input1, outputs=x)

  # --- core model on partition 2
  input2 = keras.Input(shape=(26,26,32,), name='split_layer1_output')
  input3 = keras.Input(shape=(26,26,32,), name='split_layer2_output')

  x = keras.layers.Concatenate(name='concat')([input2, input3])
  x = keras.layers.Flatten(name='flatten')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  core_model2 = keras.Model(inputs=[input2, input3], outputs=outputs)
  if rank == p_0_rank:
    return core_model0
  elif rank == p_1_rank:
    return core_model1
  elif rank == p_2_rank:
    return core_model2

def multi_input_model_generator():
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input0')
  input1 = keras.Input(shape=(26,26,32,), name='input1')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', name='conv')(input0)
  y = pgen.SplitLayer(name="split_layer0")(x)
  z = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', name='conv2')(y)
  x = pgen.SplitLayer(name="split_layer2")(x)

  x = keras.layers.Concatenate(name='concat')([input1, x, z])
  x = keras.layers.Flatten(name='flatten')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  model = keras.Model(inputs=[input0, input1], outputs=outputs)
  return model

def multi_input_partition_info(ref_model, rank):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo('0')
    in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]

    out_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    out_edge_1 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0, out_edge_1]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo('1')
    in_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('split_layer0').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0]

    out_edge_0 = pinfo.EndpointInfo(2, ref_model.get_layer('split_layer2').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0]

  elif rank == p_2_rank:
    partition_info = pinfo.PartitionInfo('2')
    in_0 = pinfo.EndpointInfo(1, ref_model.inputs[1].shape, tf.float32)
    partition_info.real_input_infos = [in_0]

    in_edge_0 = pinfo.EndpointInfo(1, ref_model.get_layer('split_layer1').output.shape, tf.float32)
    in_edge_1 = pinfo.EndpointInfo(2, ref_model.get_layer('split_layer2').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0, in_edge_1]

    out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
    partition_info.real_output_infos = [out_0]
  return partition_info

def multi_input_partitioned_core_model(rank):
  util.set_tf_random_seed()
  # --- core model on partition 0
  input0 = keras.Input(shape=(28,28,1,), name='input0')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', name='conv')(input0)
  y = keras.layers.Layer(name="split_layer0_input")(x)
  z = keras.layers.Layer(name="split_layer1_input")(x)
  core_model0 = keras.Model(inputs=input0, outputs=[y, z])

  # --- core model on partition 1
  input2 = keras.Input(shape=(26,26,32,), name='split_layer0_output')
  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', name='conv2')(input2)
  x = keras.layers.Layer(name="split_layer2_input")(x)
  core_model1 = keras.Model(inputs=input2, outputs=x)

  # --- core model on partition 2
  input1 = keras.Input(shape=(26,26,32,), name='input1')
  input3 = keras.Input(shape=(26,26,32,), name='split_layer1_output')
  input4 = keras.Input(shape=(26,26,32,), name='split_layer2_output')
  x = keras.layers.Concatenate(name='concat')([input1, input3, input4])
  x = keras.layers.Flatten(name='flatten')(x)
  outputs = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  core_model2 = keras.Model(inputs=[input1, input3, input4], outputs=outputs)
  if rank == p_0_rank:
    return core_model0
  elif rank == p_1_rank:
    return core_model1
  elif rank == p_2_rank:
    return core_model2

def multi_output_model_generator():
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Flatten(name='flatten')(input0)
  x = keras.layers.Dense(10, activation='relu', name='dense_relu')(x)
  y = pgen.SplitLayer(name="ten_classes")(x)
  z = pgen.SplitLayer(name="two_classes")(x)
  x = keras.layers.Add(name='add')([y,z])
  output0 = keras.layers.Dense(10, activation='relu', name='dense_softmax10')(x)
  output1 = keras.layers.Dense(2, activation='softmax', name='dense_softmax2')(x)
  model = keras.Model(inputs=input0, outputs=[output0, output1], name="model")
  return model

def multi_output_partition_info(ref_model, rank):
  if rank == p_0_rank:
    partition_info = pinfo.PartitionInfo('0')

    in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
    partition_info.real_input_infos = [in_0]
    out_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('ten_classes').output.shape, tf.float32)
    out_edge_1 = pinfo.EndpointInfo(1, ref_model.get_layer('two_classes').output.shape, tf.float32)
    partition_info.edge_output_infos = [out_edge_0, out_edge_1]

  elif rank == p_1_rank:
    partition_info = pinfo.PartitionInfo('1')
    in_edge_0 = pinfo.EndpointInfo(0, ref_model.get_layer('ten_classes').output.shape, tf.float32)
    in_edge_1 = pinfo.EndpointInfo(1, ref_model.get_layer('two_classes').output.shape, tf.float32)
    partition_info.edge_input_infos = [in_edge_0, in_edge_1]

    out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
    out_1 = pinfo.EndpointInfo(1, ref_model.outputs[1].shape, tf.float32)
    partition_info.real_output_infos = [out_0, out_1]
  return partition_info

def multi_output_partitioned_core_model(rank):
  util.set_tf_random_seed()
  # --- core model on partition 0
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Flatten(name='flatten')(input0)
  x = keras.layers.Dense(10, activation='relu', name='dense_relu')(x)
  y = keras.layers.Layer(name="ten_classes_input")(x)
  z = keras.layers.Layer(name="two_classes_input")(x)
  core_model0 = keras.Model(inputs=input0, outputs=[y, z])

  # --- core model on partition 1
  input1 = keras.Input(shape=(10,), name='ten_classes_output')
  input2 = keras.Input(shape=(10,), name='two_classes_output')
  x = keras.layers.Add(name='add')([input1, input2])
  output0 = keras.layers.Dense(10, activation='relu', name='dense_softmax10')(x)
  output1 = keras.layers.Dense(2, activation='softmax', name='dense_softmax2')(x)
  core_model1 = keras.Model(inputs=[input1, input2], outputs=[output0, output1])

  if rank == p_0_rank:
    return core_model0
  elif rank == p_1_rank:
    return core_model1

def simple_model_generator():
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Flatten(name='flatten')(input0)
  x = keras.layers.Dense(10, activation='softmax', name='dense_softmax')(x)
  model = keras.Model(inputs=input0, outputs=x)
  return model

def simple_partition_info(ref_model, rank):
  partition_info = pinfo.PartitionInfo('0')

  in_0 = pinfo.EndpointInfo(0, ref_model.inputs[0].shape, tf.float32)
  partition_info.real_input_infos = [in_0]
  out_0 = pinfo.EndpointInfo(0, ref_model.outputs[0].shape, tf.float32)
  partition_info.real_output_infos = [out_0]
  return partition_info

def simple_partitioned_core_model(rank):
  return simple_model_generator()

