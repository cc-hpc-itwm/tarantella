import tarantella as tnt

import numpy as np
import pytest

class TestTensorBroadcaster:

  @pytest.mark.parametrize("array_shape", [(1), (5,7), (3,2,4)])
  def test_single_array(self, array_shape):
    np.random.seed(42)
    input_array = np.random.random_sample(array_shape).astype('float32')
    rank = tnt.get_rank()
    root_rank = tnt.get_size() - 1
    broadcaster = tnt.TensorBroadcaster(input_array, root_rank)

    expected_output_array = input_array
    if rank == root_rank:
      output_array = broadcaster.broadcast(input_array)
    else:
      output_array = broadcaster.broadcast()

    result = (output_array == expected_output_array).all()
    assert isinstance(output_array, np.ndarray)
    assert result

  @pytest.mark.parametrize("list_length", [2, 3])
  @pytest.mark.parametrize("array_shape", [(1), (5,7), (3,2,4)])
  def test_list_of_arrays(self, array_shape, list_length):
    np.random.seed(42)
    input_array = np.random.random_sample(array_shape).astype('float32')
    inputs = list_length * [input_array]
    rank = tnt.get_rank()
    root_rank = 0
    broadcaster = tnt.TensorBroadcaster(inputs, root_rank)

    expected_output_array = input_array
    if rank == root_rank:
      outputs = broadcaster.broadcast(inputs)
    else:
      outputs = broadcaster.broadcast()

    result = all((array == expected_output_array).all() for array in outputs)
    assert isinstance(outputs, list)
    assert result
