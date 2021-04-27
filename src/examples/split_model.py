import pprint

import tensorflow as tf
from tensorflow import keras

import tarantella.strategy.pipelining.rank_mapper as rank_mapper
import tarantella.strategy.pipelining.core_model_builder as core_model
import tarantella.strategy.pipelining.partition_generator as pgen

def alexnet_model_generator():
  tf.random.set_seed(42)
  inputs = keras.Input(shape=(28,28,1,), name='input')
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = pgen.SplitLayer(name="split_layer0")(x)
  x = keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
  x = pgen.SplitLayer(name="split_layer1")(x)

  y = keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(y)

  y = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(y)
  x = keras.layers.Concatenate()([x, y])
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(512, activation='relu')(x)
  outputs = keras.layers.Dense(10, activation='softmax')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


model = alexnet_model_generator()
partition_generator = pgen.GraphPartitionGenerator(model)

nranks = 3
rank_mapper = rank_mapper.RankMapper(partition_generator.get_partition_graph(), nranks)
for rank in range(nranks):
  pprint.pprint(f"Model for rank {rank}")
  core_model_builder = core_model.CoreModelBuilder(partition_generator,
                                                   rank_mapper, rank)
  model = core_model_builder.get_model()
  model.summary()

