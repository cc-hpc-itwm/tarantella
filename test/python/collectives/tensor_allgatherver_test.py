import tarantella as tnt
import utilities as util

import tensorflow as tf
import numpy as np
import math

import pytest

class TestTensorAllgatherver:
  @pytest.mark.parametrize("array_length", [2])
  def test_single_array_identical_inputs(self, array_length):
    input_array = np.ones(shape=(array_length, 1), dtype=np.float32)
    expected_output_array = np.ones(shape=(array_length * tnt.get_size(), 1), dtype=np.float32)

    allgatherver = tnt.TensorAllgatherver(input_array)
    output_array = allgatherver.allgatherv(input_array)

    assert isinstance(output_array, np.ndarray)
    assert np.array_equal(output_array, expected_output_array)

  
  @pytest.mark.parametrize("list_length", [1, 5, 12])
  def test_list_of_arrays_identical_inputs(self, list_length):
    array_length = 50
    input_array = np.ones(shape=(array_length, 1), dtype=np.float32)
    input_list = [input_array for i in range(list_length)]

    expected_output_array = np.ones(shape=(array_length * tnt.get_size(), 1), dtype=np.float32)

    allgatherver = tnt.TensorAllgatherver(input_list)
    output_list = allgatherver.allgatherv(input_list)

    assert isinstance(output_list, list)
    assert all(np.array_equal(array, expected_output_array) for array in output_list)