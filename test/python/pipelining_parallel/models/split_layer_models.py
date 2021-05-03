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
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = pgen.SplitLayer(name="split_layer0")(x)

  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(512, activation='relu')(x)
  outputs = keras.layers.Dense(10, activation='softmax')(x)
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

def fc_model_generator():
  tf.random.set_seed(42)
  reference_input = keras.Input(shape=(28,28,1,), name='reference_input')
  reference_x = keras.layers.Flatten()(reference_input)
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

def skip_connection_model_generator():
  util.set_tf_random_seed()
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(inputs)
  y = pgen.SplitLayer(name="split_layer0")(x)
  z = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', activation='relu')(y)
  x = pgen.SplitLayer(name="split_layer2")(x)

  x = keras.layers.Concatenate()([x, z])
  x = keras.layers.Flatten()(x)
  outputs = keras.layers.Dense(10, activation='softmax')(x)
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

def multi_input_model_generator():
  util.set_tf_random_seed()
  input0 = keras.Input(shape=(28,28,1,), name='input0')
  input1 = keras.Input(shape=(26,26,32,), name='input1')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(input0)
  y = pgen.SplitLayer(name="split_layer0")(x)
  z = pgen.SplitLayer(name="split_layer1")(x)

  x = keras.layers.Conv2D(32, 1, strides=(1, 1), padding='valid', activation='relu')(y)
  x = pgen.SplitLayer(name="split_layer2")(x)

  x = keras.layers.Concatenate()([input1, x, z])
  x = keras.layers.Flatten()(x)
  outputs = keras.layers.Dense(10, activation='softmax')(x)
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
