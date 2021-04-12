import tarantella as tnt

import numpy as np
import pytest

class TestTensorAllreducer:
  @pytest.mark.parametrize("array_length", [8, 35, 67])
  def test_single_array_identical_inputs(self, array_length):
    input_array = np.empty(shape=(array_length, 1), dtype=np.float32)
    input_array.fill(1.0)

    expected_output_array = input_array * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    result = np.array_equal(output_array, expected_output_array)
    assert isinstance(output_array, np.ndarray) and result

  @pytest.mark.parametrize("array_length", [11, 44, 81])
  def test_single_array_different_inputs(self, array_length):
    input_array = np.empty(shape=(array_length, 1), dtype=np.float32)
    input_array.fill(tnt.get_rank())

    expected_output_array = np.empty(input_array.shape, dtype=np.float32)
    expected_output_array.fill(sum(range(tnt.get_size())))

    allreducer = tnt.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    result = np.array_equal(output_array, expected_output_array)
    assert isinstance(output_array, np.ndarray) and result

  @pytest.mark.parametrize("length", [18, 63, 99])
  def test_list_of_arrays_identical_inputs(self, length):
    input_array = np.empty(shape=(length, 1), dtype=np.float32)
    input_array.fill(1.0)
    input_list = [input_array for i in range(length)]

    expected_output_array = input_array * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_list)
    output_list = allreducer.allreduce(input_list)

    result = all(np.array_equal(arr, expected_output_array) for arr in output_list)
    assert isinstance(output_list, list) and result

  @pytest.mark.parametrize("length", [14, 44, 91])
  def test_list_of_arrays_identical_inputs_diff_types(self, length):
    input_array = np.empty(shape=(length, 1), dtype=np.float32)
    input_array.fill(1.0)
    input_list = [input_array]

    input_array_double = np.empty(shape=(length, 1), dtype=np.double)
    input_array_double.fill(1.0)
    input_list.append(input_array_double)

    expected_output_array = input_array_double * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_list)
    output_list = allreducer.allreduce(input_list)

    result = all(np.array_equal(arr, expected_output_array) for arr in output_list)
    assert isinstance(output_list, list) and result

  @pytest.mark.parametrize("length", [5, 56, 89])
  def test_dict_idential_inputs(self, length):
    input_dict = dict.fromkeys(("key " + str(i) for i in range(length)), 1.0)

    expected_output_value = 1.0 * tnt.get_size()

    allreducer = tnt.TensorAllreducer(input_dict)
    output_dict = allreducer.allreduce(input_dict)

    result = (v == expected_output_value for v in output_dict.values())
    assert isinstance(output_dict, dict) and result

  @pytest.mark.parametrize("length", [1, 22])
  def test_dict_varying_values(self, length):
    input_dict = dict()

    input_array = np.empty(shape=(length, length), dtype=np.float32)
    input_array.fill(tnt.get_rank())
    input_list = [input_array for i in range(length)]

    expected_output_array = np.empty(input_array.shape, dtype=np.float32)
    expected_output_array.fill(sum(range(tnt.get_size())))

    input_3D_array = np.empty(shape=(length, length, length), dtype=np.float32)
    input_3D_array.fill(tnt.get_rank())

    expected_output_3D_array = np.empty(input_3D_array.shape, dtype=np.float32)
    expected_output_3D_array.fill(sum(range(tnt.get_size())))

    input_dict["list_of_tensors"] = input_list
    input_dict["single_tensor"] = input_3D_array

    allreducer = tnt.TensorAllreducer(input_dict)
    output_dict = allreducer.allreduce(input_dict)

    assert all(np.array_equal(arr, expected_output_array) for arr in output_dict["list_of_tensors"])
    assert np.array_equal(output_dict["single_tensor"], expected_output_3D_array)
    assert isinstance(output_dict, dict)

  def test_single_value(self):
    input = float(tnt.get_rank())
    expected_output = sum(range(tnt.get_size()))

    allreducer = tnt.TensorAllreducer(input)
    output = allreducer.allreduce(input)

    assert (expected_output == output) and isinstance(output, float)

  def test_single_array_empty(self):
    input_array = np.empty(0, dtype=np.float32)

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(input_array)

  def test_list_empty(self):
    input_list = []

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(input_list)

  def test_unsupported_type(self):
    input = "sample input"

    with pytest.raises(TypeError):
      tnt.TensorAllreducer(input)
