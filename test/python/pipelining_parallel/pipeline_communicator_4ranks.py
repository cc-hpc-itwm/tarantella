import tarantella as tnt

import numpy as np
import pytest

import tarantella.strategy.pipelining.connection_info as cinfo

# Assumption: the partitioning graph is oriented,
# such that each rank receives data from smaller ranks and sends to larger ranks
partition_test_cases = [
  # Test case 0:
  # (rank) --conn_id-- (rank)
  # (0) --0--> (1) --1--> (2) --2--> (3)
  { # conn_id, cinfo.ConnectionInfo(<rank, rank>,  size)
            0: cinfo.ConnectionInfo((0,       1),     4),
            1: cinfo.ConnectionInfo((1,       2),    32),
            2: cinfo.ConnectionInfo((2,       3),   100),
  },
  # Test case 1:
  # (0) --3--> (2)
  #             ^
  #             |
  #             2
  #             |
  #            (1) --1--> (3)
  { # conn_id, cinfo.ConnectionInfo(<rank, rank>,  size)
            1: cinfo.ConnectionInfo((3,       1),  1000),
            2: cinfo.ConnectionInfo((1,       2),    64),
            3: cinfo.ConnectionInfo((2,       0),    68),
  },
  # Test case 2:
  # (0) --0--> (3)
  # (1) --1--> (3)
  # (2) --2--> (3)
  { # conn_id, cinfo.ConnectionInfo(<rank, rank>,  size)
            1: cinfo.ConnectionInfo((0,       3),     4),
            2: cinfo.ConnectionInfo((1,       3),     4),
            3: cinfo.ConnectionInfo((2,       3),     8),
  },
  # Test case 3:
  # (0) --0--> (3)
  # (1)
  # (2)
  { # conn_id, cinfo.ConnectionInfo(<rank, rank>,  size)
            0: cinfo.ConnectionInfo((0,       3),     4),
  },
  ]

@pytest.mark.min_tfversion('2.2')
class TestPipelineCommunicator:

  @pytest.mark.parametrize("partition", partition_test_cases)
  @pytest.mark.parametrize("num_micro_batches", [1,2,3])
  def test_send_all_connections(self, partition, num_micro_batches):
    elem_type = np.dtype(np.float32)
    pipeline_comm = tnt.PipelineCommunicator(partition, num_micro_batches)

    micro_batch_size = 4
    pipeline_comm.setup_infrastructure(micro_batch_size)

    # send on all connections
    for micro_batch_id in range(num_micro_batches):
      for conn_id in pipeline_comm.get_local_connection_ids():
        conn_info = partition[conn_id]
        if conn_info.get_other_rank(tnt.get_rank()) < tnt.get_rank():
          continue

        array_length = micro_batch_size * conn_info.get_size_in_bytes() // elem_type.itemsize
        input_array = np.empty(shape=(array_length, 1), dtype=elem_type)
        input_array.fill(tnt.get_rank())

        pipeline_comm.send(input_array,
                           connection_id = conn_id,
                           micro_batch_id = micro_batch_id)

    # receive on all connections
    for micro_batch_id in range(num_micro_batches):
      for conn_id in pipeline_comm.get_local_connection_ids():
        conn_info = partition[conn_id]
        if conn_info.get_other_rank(tnt.get_rank()) > tnt.get_rank():
          continue

        array_length = micro_batch_size * conn_info.get_size_in_bytes() // elem_type.itemsize
        expected_array = np.empty(shape=(array_length, 1), dtype=elem_type)
        expected_array.fill(conn_info.get_other_rank(tnt.get_rank()))

        input_array = np.empty(shape=(array_length, 1))
        result = pipeline_comm.recv(input_array,
                                    connection_id = conn_id,
                                    micro_batch_id = micro_batch_id)
        assert np.allclose(result, expected_array, atol=1e-6)

