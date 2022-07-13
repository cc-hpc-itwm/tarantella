import tarantella as tnt
import tensorflow as tf
import utilities as util

import enum
import numpy as np
import pytest


def build_model(ninputs: int) -> tf.keras.Model:
  util.set_tf_random_seed()
  input0 = tf.keras.Input(shape=(28,28,1,), name='i0')
  inputs = [input0]
  for i in range(1, ninputs):
    inputs += [tf.keras.Input(shape=(5,), name = f"i{i}")]
  x = tf.keras.layers.Flatten()(input0)
  if ninputs > 1:
    x = tf.keras.layers.Concatenate(axis=1)([x] + inputs[1:])
  out = tf.keras.layers.Dense(10, activation='softmax', name='softmax')(x)
  return tf.keras.Model(inputs=inputs, outputs=out)

class InputType(enum.Flag):
  UNIQUE = enum.auto()
  LIST = enum.auto()
  DICT = enum.auto()

def build_dataset(input_type: InputType, ninputs: int) -> tf.data.Dataset:
  i0 = tf.data.Dataset.from_tensor_slices(np.ones((32,28,28,1)), name = 'i0')
  labels = tf.data.Dataset.from_tensor_slices(np.ones((32,10)))

  if input_type == InputType.UNIQUE:
    inputs = i0
  else:
    inputs = [i0]
    for i in range(1, ninputs):
      inputs += [tf.data.Dataset.from_tensor_slices(np.ones((32, 5)), name = f"i{i}")]

    inputs = tuple(inputs)
    if input_type == InputType.LIST:
      inputs = tf.data.Dataset.zip(inputs)
    if input_type == InputType.DICT:
      inputs = tf.data.Dataset.zip(inputs).map(lambda *inputs:
                                               {f"i{i}": inputs[i] for i in range(len(inputs))})
  return tf.data.Dataset.zip((inputs, labels)).batch(4)


class TestCloneModel:
  @pytest.mark.parametrize("input_type_and_count", [(InputType.UNIQUE, 1),
                                                    (InputType.LIST, 2),
                                                    (InputType.LIST, 3),
                                                    (InputType.DICT, 1),
                                                    (InputType.DICT, 2),
                                                    ])
  @pytest.mark.parametrize("parallel_strategy", [tnt.ParallelStrategy.DATA,
                                                 pytest.param(tnt.ParallelStrategy.ALL, marks=pytest.mark.skip),
                                                 ])
  def test_clone_keras_model(self, input_type_and_count, parallel_strategy):
    input_type, ninputs = input_type_and_count
    metric = 'mean_squared_error'

    ref_model = build_model(ninputs=ninputs)
    ref_model.compile(loss='mse', metrics = metric)
    dataset = build_dataset(input_type, ninputs = ninputs)
    ref_history = ref_model.fit(dataset)

    model = build_model(ninputs=ninputs)
    tnt_model = tnt.Model(model, parallel_strategy = parallel_strategy)
    tnt_model.compile(loss='mse', metrics = metric)
    dataset = build_dataset(input_type, ninputs = ninputs)
    tnt_history = tnt_model.fit(dataset)

    result = [True, True]
    if tnt.is_master_rank():
      result = [np.allclose(tnt_history.history['loss'], ref_history.history['loss'], atol=1e-4),
                np.allclose(tnt_history.history[metric], ref_history.history[metric], atol=1e-6)]
    util.assert_on_all_ranks(result)

