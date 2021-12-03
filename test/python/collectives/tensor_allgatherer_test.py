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
    expected_output_array = np.ones(shape=(array_length * tnt.get_size()), dtype=np.float32)

    allgatherer = tnt.TensorAllgatherer(input_array)
    output_array = allgatherer.allgather(input_array)

    assert isinstance(output_array, np.ndarray)
    assert np.array_equal(output_array, expected_output_array)

  
  def test_scalar_identical_inputs(self):
    scalar = 50

    expected_output_array = np.ones(shape=(tnt.get_size()), dtype=int)
    expected_output_array.fill(scalar)

    allgatherer = tnt.TensorAllgatherer(scalar)
    output_array = allgatherer.allgather(scalar)

    assert isinstance(output_array, np.ndarray)
    assert np.array_equal(output_array, expected_output_array)