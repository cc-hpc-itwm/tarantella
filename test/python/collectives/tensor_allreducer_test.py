import tarantella as tnt
import utilities as util

import tensorflow as tf
import numpy as np
import math

import pytest

class TestTensorAllreducer:
  @pytest.mark.parametrize("array_length", [8, 35, 67])
  @pytest.mark.parametrize("dtype", [np.float32, np.int32])
  def test_single_array_identical_inputs(self, array_length, dtype):
    input_array = np.ones(shape=(array_length, 1), dtype=dtype)
    expected_output_array = input_array * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    assert isinstance(output_array, np.ndarray)
    assert np.array_equal(output_array, expected_output_array)

  @pytest.mark.parametrize("array_length", [11, 44, 81])
  def test_single_array_different_inputs(self, array_length):
    input_array = np.empty(shape=(array_length, 1), dtype=np.float32)
    input_array.fill(tnt.get_rank())

    expected_output_array = np.empty(input_array.shape, dtype=np.float32)
    expected_output_array.fill(sum(range(tnt.get_size())))

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    assert isinstance(output_array, np.ndarray)
    assert np.array_equal(output_array, expected_output_array)

  @pytest.mark.parametrize("list_length", [1, 5, 12])
  @pytest.mark.parametrize("dtype", [np.float32, np.int32])
  def test_list_of_arrays_identical_inputs(self, list_length, dtype):
    array_length = 50
    input_array = np.ones(shape=(array_length, 1), dtype=dtype)
    input_list = [input_array for i in range(list_length)]

    expected_output_array = input_array * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_list)
    output_list = allreducer.allreduce(input_list)

    assert isinstance(output_list, list)
    assert all(np.array_equal(array, expected_output_array) for array in output_list)

  def test_list_of_arrays_identical_inputs_diff_types(self):
    input_array_float = np.ones(shape=(238, 1), dtype=np.float32)
    input_array_double = np.ones(shape=(42, 1), dtype=np.double)
    another_input_array_float = np.ones(shape=(99, 1), dtype=np.float32)
    input_list = [input_array_float, input_array_double, another_input_array_float]

    expected_output_list = [array * tnt.get_size() for array in input_list]

    allreducer = tnt.TensorAllreducer(input_list)
    output_list = allreducer.allreduce(input_list)

    assert isinstance(output_list, list)
    assert all(np.array_equal(output_array, expected_output_array) \
               for (output_array, expected_output_array) \
               in zip(output_list, expected_output_list))

  @pytest.mark.parametrize("length", [5, 56, 89])
  def test_dict_many_keys(self, length):
    input_value = 4.2
    input_dict = dict.fromkeys(("key " + str(i) for i in range(length)), input_value)
    expected_output_value = input_value * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_dict)
    output_dict = allreducer.allreduce(input_dict)

    assert isinstance(output_dict, dict)
    assert len(input_dict) == len(output_dict)
    assert all(v == expected_output_value for v in output_dict.values())

  def test_dict_varying_values(self):
    value1 = 3.29
    value2 = 17.0

    input_2D_array = np.full(shape=(4, 5), fill_value=value1, dtype=np.float32)
    input_list = [input_2D_array, input_2D_array, input_2D_array]
    input_3D_array = np.full(shape=(2, 15, 4), fill_value=value2, dtype=np.float64)

    input_dict = dict()
    input_dict["list_of_tensors"] = input_list
    input_dict["single_tensor"] = input_3D_array

    expected_output_2D_array = tnt.get_size() * input_2D_array
    expected_output_3D_array = tnt.get_size() * input_3D_array

    allreducer = tnt.TensorAllreducer(input_dict)
    output_dict = allreducer.allreduce(input_dict)

    assert isinstance(output_dict, dict)
    assert len(output_dict) == 2
    assert len(output_dict["list_of_tensors"]) == 3
    assert all(np.array_equal(array, expected_output_2D_array) for array in output_dict["list_of_tensors"])
    assert np.array_equal(output_dict["single_tensor"], expected_output_3D_array)

  def test_single_value(self):
    inputs = float(tnt.get_rank())
    expected_output = sum(range(tnt.get_size()))

    allreducer = tnt.TensorAllreducer(inputs)
    output = allreducer.allreduce(inputs)

    assert isinstance(output, float)
    assert expected_output == output

  def test_single_array_empty(self):
    input_array = np.empty(shape=0, dtype=np.float32)

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(input_array)

  def test_list_empty(self):
    input_list = []

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(input_list)

  def test_unsupported_type(self):
    string = "sample input"

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(string)

  @pytest.mark.parametrize("input_value, dtype", [(4.67, np.float32),
                                                  (4.00000067, np.double),
                                                  (-42111, np.int32),
                                                  (42, np.int16)])
  def test_tensor_numeric(self, input_value, dtype):
    expected_value = input_value * tnt.get_size()
    input = tf.constant(input_value, dtype=dtype)

    allreducer = tnt.TensorAllreducer(input)
    output = allreducer.allreduce(input)

    assert tf.is_tensor(output)
    assert output == expected_value

  @pytest.mark.parametrize("input_shape", [(6,), (11, 12), (4, 5, 6)])
  @pytest.mark.parametrize("dtype", [np.float32, np.int32])
  def test_nd_tensor(self, input_shape, dtype):
    input_array = np.ones(shape=input_shape, dtype=dtype)
    expected_output_array = input_array * tnt.get_size()

    input = tf.constant(input_array)

    allreducer = tnt.TensorAllreducer(input)
    output = allreducer.allreduce(input)

    assert tf.is_tensor(output)
    assert np.array_equal(output.numpy(), expected_output_array)

  @pytest.mark.parametrize("input_dict", [{ "a" : tf.constant(22.610077, dtype=np.float32),
                                            "b" : tf.constant(1.000610077, dtype=np.float64)},
                                          { "a" : tf.constant(-31222, dtype=np.int16),
                                            "b" : tf.constant(55000111, dtype=np.int32)}
                                          ])
  def test_dict_of_tensors(self, input_dict):
    expected_dict = { k: v * tnt.get_size() for k, v in input_dict.items() }

    allreducer = tnt.TensorAllreducer(input_dict)
    output_dict = allreducer.allreduce(input_dict)

    assert { k: tf.is_tensor(v) for k, v in output_dict.items() }
    assert output_dict == expected_dict

  @pytest.mark.parametrize("array_length", [35])
  @pytest.mark.parametrize("index", [12])
  def test_array_nan(self, array_length, index):
    injection_rank = util.same_random_int_all_ranks(0, tnt.get_size())
    input_array = np.ones(shape=(array_length, 1), dtype=np.float32)
    if tnt.get_rank() == injection_rank:
      input_array[index] = math.nan

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    assert np.isnan(output_array[index])

  @pytest.mark.parametrize("array_length", [12, 78])
  @pytest.mark.parametrize("index", [7, 3])
  def test_array_inf(self, array_length, index):
    injection_rank = util.same_random_int_all_ranks(0, tnt.get_size())
    input_array = np.ones(shape=(array_length, 1), dtype=np.float32)
    if tnt.get_rank() == injection_rank:
      input_array[index] = math.inf

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    assert np.isinf(output_array[index])
