import tarantella

import numpy as np
import pytest

class TestTensorAllreducer:
  def test_initialization(self, tarantella_framework):
    assert tarantella_framework

  @pytest.mark.parametrize("array_length", [8, 35, 67])
  def test_single_array_identical_inputs(self, tarantella_framework, array_length):
    input_array = np.empty(shape=(array_length, 1), dtype=np.float32)
    input_array.fill(1.0)

    expected_output_array = input_array * tarantella_framework.get_size()

    allreducer = tarantella_framework.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    result = (output_array == expected_output_array)
    assert isinstance(output_array, np.ndarray) and result.all()

  @pytest.mark.parametrize("array_length", [11, 44, 81])
  def test_single_array_different_inputs(self, tarantella_framework, array_length):
    input_array = np.empty(shape=(array_length, 1), dtype=np.float32)
    input_array.fill(tarantella_framework.get_rank())

    expected_output_array = np.empty(input_array.shape, dtype=np.float32)
    expected_output_array.fill(sum(range(tarantella_framework.get_size())))

    allreducer = tarantella_framework.TensorAllreducer(input_array)
    output_array = allreducer.allreduce(input_array)

    result = (output_array == expected_output_array)
    assert isinstance(output_array, np.ndarray) and result.all()

  @pytest.mark.parametrize("length", [18, 63, 99])
  def test_list_of_arrays_identical_inputs(self, tarantella_framework, length):
    input_array = np.empty(shape=(length, 1), dtype=np.float32)
    input_array.fill(1.0)
    input_list = [input_array for i in range(length)]

    expected_output_array = input_array * tarantella_framework.get_size()

    allreducer = tarantella_framework.TensorAllreducer(input_list)
    output_list = allreducer.allreduce(input_list)

    result = all((arr == expected_output_array).all() for arr in output_list)
    assert isinstance(output_list, list) and result

  def test_single_array_empty(self, tarantella_framework):
    input_array = np.empty(0, dtype=np.float32)

    with pytest.raises(TypeError):
      tarantella_framework.TensorAllreducer(input_array)

  def test_list_empty(self, tarantella_framework):
    input_list = []

    with pytest.raises(TypeError):
      tarantella_framework.TensorAllreducer(input_list)

  def test_unsupported_type(self, tarantella_framework):
    input = "sample input"

    with pytest.raises(TypeError):
      tarantella_framework.TensorAllreducer(input)